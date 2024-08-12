#!/usr/bin/env python3

import os,sys

Eval = False

if len(sys.argv)>1:
  if sys.argv[1].lower()=='eval':
    Eval=True
  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES']=""

# Only use one GPU for evaluations
if Eval:
  os.environ['CUDA_VISIBLE_DEVICES']="1"
  
import tensorflow as     tf
import tensorflow.keras.mixed_precision as mixed_precision
#import tensorflow.keras.backend as K
import tensorflow_probability as tfp
import copy
import numpy as np 
import matplotlib.pyplot as plt
import time
import logging
from MakeTraining_tf_v2 import *
from FIDNet import *

#print( tf.config.list_physical_devices('GPU') )
#sys.exit(10)

#Suppress warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
#
# Powers (momenta) to use to restrain the normal distributions of diff/sigma (chi2)
powers = tf.constant([0.5,2.0,3.0,3.5], dtype=tf.complex128)
#powers = tf.cast(tf.linspace(0.5,4.0,97),tf.complex128) # < used to pick subsets
normal_momenta = tf.random.normal(mean=0.0, stddev=1.0, shape=(16*400*512,1))
normal_momenta = tf.cast( normal_momenta, tf.complex128 )
normal_momenta = tf.math.reduce_mean(tf.pow(normal_momenta, tf.reshape(powers,(1,-1))), axis=0)

StatusFile = sys.argv[0].replace('py','status')
#
# Good hyper parameters
if Eval:
  ROUNDS     =  1
else:
  ROUNDS     = 10000
EPOCHS             = 1
BATCHSIZE          = 1 #24
fidnet_filters_1h  = 20
fidnet_filters_13c = 20
checkpoint_path = "./checkpointer/fidnet_model_7pts/"

#
# Distributed learning
strategy = tf.distribute.MirroredStrategy()
#
# Initiate mixed precision
#if (not Eval) and False:
#  policy = mixed_precision.Policy('mixed_float16')
#  mixed_precision.set_global_policy(policy)
#
# Learning rate schedules 
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model=20*20*13, warmup_steps=20000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):  # Step is batch
    step = step 
    arg1 = tf.math.rsqrt( tf.cast(step, tf.float32) )
    arg2 = tf.cast(step, tf.float32) * (self.warmup_steps ** -1.5)
    val = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2) / 40. 
    return val

with strategy.scope():
  learning_rate = CustomSchedule()

  optimizer = tf.keras.optimizers.legacy.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
  optimizer = mixed_precision.LossScaleOptimizer( optimizer )

  @tf.function
  def loss_function(real, pred):
    vloss = loss_value(real,pred)
    sloss = loss_sigma_3(real,pred)
    tloss = loss_sigma_4(real,pred)*5.
    #
    # we need to ensure no underflow with tf.math.min()
    scaling = tf.math.sigmoid( -tf.math.exp( tf.math.minimum(vloss,0.03) * 100. ))/0.268
    #
    ##return ( 2.*vloss + 0.01)/(vloss + 0.01) * vloss + 0.01/(vloss + 0.01 )*sloss
    return (2. - scaling )*vloss + scaling * ( sloss + tloss )

  @tf.function
  def loss_sigma_4(real,pred):
    pred_value = pred[...,0]
    pred_sigma = pred[...,1] # sigma
    #
    sigma_list= tf.math.scalar_mul(1.0, pred_sigma )  # (15,400,512)
    diff_list = real - pred_value
    #
    # Reshape 
    sigma_list = tf.reshape( sigma_list, (-1,1))
    diff_list  = tf.reshape( diff_list,  (-1,1))
    #chi2_list = tf.reshape( diff_list/sigma_list, (-1,1))
    #
    sigma_parts = tf.reshape( tfp.stats.quantiles( sigma_list, 200 ), (1,-1))
    #
    # We scale to reweight the contribution more towards the larger sigmas
    #norm_sigma_parts =  (sigma_parts - tf.math.reduce_min(sigma_parts))/(tf.math.reduce_max(sigma_parts)-tf.math.reduce_min(sigma_parts))
    #norm_sigma_parts = tf.math.pow( norm_sigma_parts, 0.90 ) # [0:1]

    #rw_sigma_parts = norm_sigma_parts * (tf.math.reduce_max(sigma_parts)-tf.math.reduce_min(sigma_parts)) + tf.math.reduce_min(sigma_parts)
    #sigma_parts = rw_sigma_parts
    #
    # Indices with sigma parts
    idxs = tf.where( ( sigma_parts[:,:-1] < sigma_list ) & \
                     ( sigma_parts[:,1:]  > sigma_list ), tf.constant(1.,dtype=tf.float32) , tf.constant(0.,dtype=tf.float32) ) #(5*400*512,sigma_parts)

    mean_diff_list  = tf.math.reduce_mean(diff_list)
    diff_list_tiles = tf.tile( diff_list, (1,sigma_parts.shape[-1]-1)) # (:, sigma_parts)
    counts = tf.math.reduce_sum( idxs, axis=0, keepdims=True ) #(1,sigma_parts)
    #
    # 
    calc_sigma = tf.math.reduce_sum( tf.square( (diff_list_tiles - mean_diff_list) * idxs ), axis=0, keepdims=True)/counts
    calc_sigma = tf.math.sqrt( calc_sigma )
    #
    pred_sigma = tf.math.reduce_sum( sigma_list * idxs, axis=0, keepdims=True )/counts  # (-1,1) * (-1,100) / (1,100)
    #
    sigma_loss_4 = tf.reduce_mean(tf.square(pred_sigma - calc_sigma))
    #sigma_loss_4 = tf.math.scalar_mul(0.1, sigma_loss_4)
    #sigma_loss_4 = tf.square( 0.5*(sigma_parts[0,1:]+sigma_parts[0,:-1]) - (calc_sigma[0,:] * 0.5*(sigma_parts[0,1:]+sigma_parts[0,:-1])) )
    #sigma_loss_4 = 0.25*tf.reduce_mean(sigma_loss_4)
    #
    return sigma_loss_4

  @tf.function
  def loss_sigma_3(real,pred):
    pred_value = pred[...,0]
    pred_sigma = pred[...,1] # sigma
    #
    sigma_list= tf.math.scalar_mul(1.0, pred_sigma )  # (15,400,512)
    diff_list = real - pred_value
    #
    # Reshape 
    sigma_list = tf.reshape( sigma_list, (-1,1))
    diff_list  = tf.reshape( diff_list,  (-1,1))
    #
    # number of quantiles of sigma.
    # This makes sure that both small and large sigmas will have a chi2
    # distribution that is Gaussian
    sigma_parts = tf.reshape( tfp.stats.quantiles( sigma_list, 20 ), (1,-1)) 
    #
    # Indices with sigma parts
    idxs = tf.where( ( sigma_parts[:,:-1] < sigma_list ) & \
                     ( sigma_parts[:,1:]  > sigma_list ), tf.constant(1.,dtype=tf.float64) , tf.constant(0.,dtype=tf.float64) ) #(5*400*512,10)
    #idxs = tf.ones(dtype=tf.float64, shape=sigma_list.shape)
    #
    # YAK - we need to work in float64 .. 
    chi2 =      tf.cast(diff_list/sigma_list, tf.float64) # (5*400*512,1)
    #
    chi2 =      tf.tile(chi2, (1,sigma_parts.shape[-1]-1))
    mean_chi2 = tf.math.reduce_mean(chi2)

    counts = tf.math.reduce_sum( idxs, axis=0, keepdims=True ) #(1,10)
    counts = tf.expand_dims( tf.expand_dims( counts, axis=0), axis=-1 )
    idxs   = tf.expand_dims( tf.expand_dims( idxs, axis=0), axis=-1 )
        
    chi2      = tf.complex( chi2,       tf.constant(0.,dtype=tf.float64) ) #tf.cast( chi2, tf.complex128 )
    mean_chi2 = tf.complex( mean_chi2,  tf.constant(0.,dtype=tf.float64) ) #tf.cast( mean_chi2, tf.complex128)
    idxs      = tf.complex( idxs,       tf.constant(0.,dtype=tf.float64) ) #tf.cast( idxs, tf.complex128 )
    counts    = tf.complex( counts,     tf.constant(0.,dtype=tf.float64) ) #tf.cast( counts, tf.complex128 )

    # This does not seem very stable ? 
    #power_idx = tf.random.uniform(minval=0,maxval=97,shape=(3,),dtype=tf.int32)
    #this_powers = tf.gather(powers,power_idx)
    #this_normal_momenta = tf.expand_dims( tf.gather(normal_momenta,power_idx), axis=0)
    this_powers = powers
    this_normal_momenta = tf.expand_dims( normal_momenta, axis=0)

    sigma_loss_3c = tf.pow( tf.expand_dims( chi2 - mean_chi2, axis=-1), tf.reshape(this_powers, (1,1,-1)))  # (:, sigma_parts ,powers)
    sigma_loss_3c = tf.expand_dims( sigma_loss_3c, axis=0) # (1,:,sigma,powers)
    sigma_loss_3c = sigma_loss_3c * idxs
    sigma_loss_3c = tf.math.reduce_sum( sigma_loss_3c/counts, axis=(0,1)) #(sigma,powers)
    #sigma_loss_3c = tf.pow( sigma_loss_3c, tf.reshape( 1./powers, (1,-1)) )

    # standard square potential - random serious outliers makes this unstable
    #sigma_loss_3c = tf.cast( tf.reduce_mean( tf.square( tf.abs(this_normal_momenta - sigma_loss_3c) ) ), pred_sigma.dtype )

    # this should be more robust - using soft l1 norm
    sigma_loss_3c = tf.abs(this_normal_momenta - sigma_loss_3c)
    sigma_loss_3c = tf.cast( sigma_loss_3c, pred_sigma.dtype )

    sigma_loss_3c_l1 = 2. * (tf.math.sqrt( 1. + sigma_loss_3c) - 1. ) # soft L1
    sigma_loss_3c_l2 = tf.square(sigma_loss_3c)                       # L2

    # soft change between L1 and L2 at ~ 0.5
    scaling = tf.math.sigmoid( ( sigma_loss_3c - 0.5)*10. )
    sigma_loss_3c = sigma_loss_3c_l2*(1. - scaling) + \
      (sigma_loss_3c_l1-0.2)*scaling
    
    sigma_loss_3c = tf.math.reduce_mean( sigma_loss_3c )

    # we also add a small push on sigma to not be too big    
    return tf.math.scalar_mul(0.1, sigma_loss_3c) #/tf.constant(2.,dtype=pred_sigma.dtype) #+ tf.math.reduce_mean(tf.pow(0.01*pred_sigma,4.))

  @tf.function
  def loss_value(real,pred):
    pred_val = pred[...,0]
    return tf.math.reduce_mean( tf.math.square( real - pred_val ))

  train_loss = tf.keras.metrics.Mean(name='train_loss')

  fidnet_2d_model = CombinedFIDNet(
    fidnet_filters_1h=fidnet_filters_1h,
    fidnet_filters_13c=fidnet_filters_13c,
    blocks=3,
    fidnet_kernel=(7,10),
    refine_kernel=(13,13),
    rate=0.10 )

  ckpt = tf.train.Checkpoint(transformer=fidnet_2d_model,
                             optimizer=optimizer)

  ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)
  #
  # if a checkpoint exists, restore the latest checkpoint.
  if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
  
    loss_array=np.load( open(checkpoint_path+'/loss.npy','rb'))
    lr_array = np.load( open(checkpoint_path+'/lr.npy', 'rb'))
                        
    loss_array = list(loss_array)
    lr_array = list(lr_array)
    print('\n !! Latest checkpoint restored!! \n')
  else:
    loss_array=[]
    lr_array=[]
    os.system('mkdir -p %s ' %(checkpoint_path))  
  #
  @tf.function
  def convolve2(spec, time, transpose=False, offset=0.40, end=0.98, power=2.0):
    if transpose:
      spec = tf.transpose( spec, perm=(0,2,1))
    
    spec = hilbert(spec)
    mypi = tf.math.asin(1.)*2.
    #
    spec_time = tf.math.conj(tf.signal.fft( spec ))/tf.complex( tf.cast( spec.shape[-1], tf.float32), 0.)

    # make TD over the batches
    #TD = tf.cast(tf.reduce_max(tf.where( time > 0., tf.range(time.shape[-1], dtype=tf.int32), 0))+1, tf.dtypes.int32)//2
    myrange = tf.range(time.shape[-1], dtype=tf.int32)
    myrange = tf.reshape( myrange, (1,-1))
    myrange = tf.tile( myrange, (time.shape[0],1))
    # TD array over batch
    TD = tf.cast(tf.reduce_max(tf.where( time[:,...] > 0., myrange, 0), axis=-1)+1, tf.dtypes.int32)//2
    #
    # let's make window.
    if offset is not None:
      myrange = tf.reshape( tf.range(time.shape[-1]//2, dtype=tf.float32), (1,-1))
      window = tf.math.pow(
          tf.math.sin(3.1415*offset + 3.1415*(end-offset)*tf.cast(myrange,tf.float32)/tf.expand_dims(tf.cast(TD,tf.float32),axis=-1))
      ,power)
    else:
      window = tf.ones(shape=(time.shape[0],time.shape[-1]//2), dtype=tf.float32)
    #
    # Zero all larger than TD
    myrange = tf.reshape( tf.range(time.shape[-1]//2, dtype=tf.int32), (1,-1))
    window = tf.where( myrange < tf.reshape(TD, (-1,1)), window, 0.)
    # zero fill
    window = tf.pad( window, [ [0,0], [0,window.shape[-1]]], "CONSTANT", constant_values=0.)
    window = tf.complex( window, 0.)
    window = tf.expand_dims( window, axis=1)
    #
    spec = tf.signal.fft( spec_time * window )
    if transpose:
      spec = tf.transpose( spec, perm=(0,2,1))
    #
    return tf.math.real(spec), TD


  @tf.function
  def convolve(spec, time, transpose=False, offset=0.40, end=0.98, power=2.0):
    if transpose:
      spec = tf.transpose( spec, perm=(0,2,1))
    spec = hilbert(spec)
    mypi = tf.math.asin(1.)*2.
    #
    TD = tf.cast(tf.reduce_max(tf.where( time > 0., tf.range(time.shape[-1], dtype=tf.int32), 0))+1, tf.dtypes.int32)//2
    spec_time = tf.math.conj(tf.signal.fft( spec ))/tf.complex( tf.cast( spec.shape[-1], tf.float32), 0.)
    if offset is not None:
      window = tf.math.pow(
        tf.math.sin(mypi*offset + mypi*(end-offset)*tf.range(TD, dtype=tf.float32)/tf.cast(TD, tf.float32))
        ,power)
    else:
      window = tf.ones(shape=(TD,), dtype=tf.float32)
    #
    window = tf.pad( window, [[0, spec_time.shape[-1] - TD]], "CONSTANT", constant_values=0.)
    window = tf.complex( tf.expand_dims( tf.expand_dims( window, axis=0 ), axis=0), 0.)
    spec = tf.signal.fft( spec_time * window )
    if transpose:
      spec = tf.transpose( spec, perm=(0,2,1))
    return tf.math.real(spec), TD
  
  class MyModel(tf.keras.Model):
    
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      #self.loss_tracker = []
      #self.loss_tracker = tf.keras.metrics.Mean(name='loss')
      # Get loss with self.loss_tracker.result()
      self.SigmaScale = tf.Variable( initial_value= tf.constant(1.0, dtype=tf.float32), name='SigmaScale', trainable=False)
            
    #@tf.function(input_signature=train_step_signature)
    def train_step(self, data):
      (time, time_in, inp_nc, inp_c), tar = data
      #
      # time    = (None,1,512)
      # time_in = (None,400)
      # inp_nc  = (None,1,400,512)
      # inp_c   = (None,1,400,512)
      # tar     = (None,400,512)
      #
      with tf.GradientTape() as tape:
        #
        #
        y_pred = self((time, time_in, inp_nc, inp_c), training=True) # (None,400,512) : 
        y_pred_spec = y_pred[...,0] # actual spectrum
        y_pred_conf = y_pred[...,1] # confidences (to be translated to sigma,esd below )
        #
        # First convolve spectra and uncertainties
        tar, _ = convolve2( tar, time_in,    transpose=True, offset=0.40)
        tar, _ = convolve2( tar, time[:,0,:],transpose=False, offset=0.40) #pts_mask
        #
        y_pred_spec, _ = convolve2( y_pred_spec, time_in,     transpose=True, offset=0.40)
        y_pred_spec, _ = convolve2( y_pred_spec, time[:,0,:], transpose=False, offset=0.40)
        #
        y_pred_conf, _ = convolve2( y_pred_conf, time_in,     transpose=True, offset=0.40)
        y_pred_conf, _ = convolve2( y_pred_conf, time[:,0,:], transpose=False, offset=0.40)
        #
        sigma = tf.math.scalar_mul(0.998, tf.math.sigmoid(y_pred_conf)) + tf.constant( 0.001, dtype=y_pred_spec.dtype)
        sigma = tf.math.reciprocal_no_nan(sigma)
        sigma = tf.math.subtract( sigma, tf.constant(1., dtype=sigma.dtype))
        sigma = tf.math.scalar_mul(0.50,sigma)
        #
        #
        y_pred = tf.stack( [y_pred_spec, sigma ], axis=-1)
        
        loss = self.compiled_loss( tf.math.scalar_mul(1.0, tar ) , \
                                   tf.math.scalar_mul(1.0, y_pred  ) )

        scaled_loss = self.optimizer.get_scaled_loss( loss )

      scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
      gradients = self.optimizer.get_unscaled_gradients( scaled_gradients )
      #gradients = tape.gradient(loss, self.trainable_variables)
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

      self.compiled_metrics.update_state( tf.math.scalar_mul(1.0, tar  ) , \
                                          tf.math.scalar_mul(1.0, y_pred ) )
      #
      return {m.name: m.result() for m in self.metrics}

    def get_config(self):
      return {}

  inputs = (tf.keras.Input(shape=(1,512)),     \
            tf.keras.Input(shape=(400,)),      \
            tf.keras.Input(shape=(1,400,512)), \
            tf.keras.Input(shape=(1,400,512))  )
  outputs = fidnet_2d_model([inputs[0], inputs[1], inputs[2], inputs[3] ])
  model = MyModel(inputs=inputs, outputs=outputs)
  model.compile(optimizer=optimizer, loss=loss_function, metrics=[loss_value, loss_sigma_3, loss_sigma_4 ] )

  number_of_signals = 10
  SharpValue=66.
  HScaling=(1.0,)
  Sparse=(0.999,0.9999)
  SigmaScale=1.0
  #
  # read status:
  if os.path.isfile(StatusFile):
    for l in open(StatusFile):
      its=l.split()
      if its[0]=='number_of_signals':
        number_of_signals = int(its[1])
      if its[0]=='SharpValue':
        SharpValue= float(its[1])
      #if its[0]=='SigmaScale':
      #  SigmaScale=float(its[1])
      if its[0]=='HScaling':
        temp=[1.0,]
        temp[0] = float(its[1])
        #temp[1] = float(its[2])
        #temp[2] = float(its[3])
        HScaling=(temp[0],)
      if its[0]=='Sparse':
        temp=[0.999,0.9999]
        temp[0] = float(its[1])
        temp[1] = float(its[2])
        Sparse=(temp[0],temp[1])
        
    print(f' INFO: Reading parameters ')
    print(f' INFO: Signals = {number_of_signals}, SharpValue = {SharpValue :5.2f} ')
    print(f' INFO: Sparse  = ({Sparse[0] :.4f},{Sparse[1] :.4f})')
    #print(f' INFO: SigmaScale  = {SigmaScale :.4f}')
    #
    # Set the variable SigmaScale
    model.SigmaScale.assign(SigmaScale)
    
  if Eval:
    # read from status file
    #number_of_signals=50
    pass

  preweight=None
  for round in range(ROUNDS):
    start = time.time()
    print(f' #\n # Start round {round + 1}\n #')
    sys.stdout.flush()
    #
    # First make training planes
    #
    nsigs = tf.random.uniform(minval=number_of_signals//5, maxval=number_of_signals, shape=(2000,), dtype=tf.dtypes.int32)  # Array with number of signals
    nsigsv = tf.random.uniform(minval=number_of_signals//5, maxval=number_of_signals, shape=(2*BATCHSIZE,), dtype=tf.dtypes.int32)
    
    if Eval:
      nsigs = nsigs[:20]
      
    ds  = tf.data.Dataset.from_tensor_slices( nsigs )
    dsv = tf.data.Dataset.from_tensor_slices( nsigsv )

    if Eval:
      ds = ds.map( lambda x: MakeTraining(NSignals=x, Noise=(2.0,1.0), Phase=(0,5.), Eval=True, Sharp1H=False, Sharp13C=True, SharpFuncs=(None, Sharp13C), HScaling=(1.,), Condense1H=True, Sparse=Sparse , Solvent=True, Roofing=0.080 ), num_parallel_calls= tf.data.experimental.AUTOTUNE )
      ds = ds.batch(2, drop_remainder=True)
    else:
      #
      def Sharp1H(x, SharpValue=SharpValue ):
        return 0.8*tf.math.tanh( x / SharpValue )*SharpValue + 0.2 * x
      #
      # Main dataset
      ds = ds.map( lambda x: MakeTraining(NSignals=x, Noise=(2.0,1.0), Phase=(0,5.), Sharp1H=False, Sharp13C=True, SharpFuncs=(None, Sharp13C), HScaling=(1.,), Condense1H=True, Sparse=Sparse, Solvent=True, Roofing=0.080 ), num_parallel_calls= tf.data.experimental.AUTOTUNE )      
      ds = ds.batch(BATCHSIZE, drop_remainder=True)
      #
      # Validation dataset
      dsv = dsv.map( lambda x: MakeTraining(NSignals=x, Noise=(2.0,1.0), Phase=(0,5.), Sharp1H=False, Sharp13C=True, SharpFuncs=(None, Sharp13C), HScaling=(1.,), Condense1H=True, Sparse=Sparse ), num_parallel_calls= tf.data.experimental.AUTOTUNE )      
      dsv = dsv.batch(BATCHSIZE, drop_remainder=True)
      
    ds  = ds.prefetch(tf.data.experimental.AUTOTUNE)
    dsv = dsv.prefetch(tf.data.experimental.AUTOTUNE)

    if Eval:
      counter=0
      acum=[]
      
      for elem in ds:
        time   =elem[0][0]
        time_in=elem[0][1]
        c=elem[0][2]
        d=elem[0][3]

        tar = elem[1] #(batch, 13C, 1H )
        
        y_pred = model([time,time_in,c,d], training=False)
        y_pred_spec = y_pred[...,0] # actual spectrum
        y_pred_conf = y_pred[...,1] # confidences (to be translated to sigma,esd below )
        #
        # First convolve spectra and uncertainties
        tar, tdc = convolve2( tar, time_in,     transpose=True, offset=0.40)
        tar, tdh = convolve2( tar, time[:,0,:], transpose=False, offset=0.40) #pts_mask
        #
        y_pred_spec, _ = convolve2( y_pred_spec, time_in,     transpose=True, offset=0.40)
        y_pred_spec, _ = convolve2( y_pred_spec, time[:,0,:], transpose=False, offset=0.40)
        #
        y_pred_conf, _ = convolve2( y_pred_conf, time_in,     transpose=True, offset=0.40)
        y_pred_conf, _ = convolve2( y_pred_conf, time[:,0,:], transpose=False, offset=0.40)
        #
        sigma = tf.math.scalar_mul(0.998, tf.math.sigmoid(y_pred_conf)) + tf.constant( 0.001, dtype=y_pred_spec.dtype)
        sigma = tf.math.reciprocal_no_nan(sigma)
        sigma = tf.math.subtract( sigma, tf.constant(1., dtype=sigma.dtype))
        sigma = tf.math.scalar_mul(0.50,sigma)
        #
        #
        y_pred = tf.stack( [y_pred_spec, sigma ], axis=-1)

        #print(f' SR    = {sr :.2f}  ', end='')
        print(f' RMSD  = {np.sqrt(np.mean(np.square(y_pred_spec[0,:,:] - tar[0,:,:]))) :13.6e} ', end='')
        print(f' LOSS  = {np.mean(np.square( y_pred_spec[0,:,:] - tar[0,:,:])) :13.6e} ', end='')
        print('')
          
        #
        print(f' loss_value: {loss_value(tar,y_pred).numpy() :.4f} loss_sigma_3: {loss_sigma_3(tar,y_pred).numpy() :.4f}' , end='')
        acum.append( loss_sigma_3(tar,y_pred).numpy())
        print(' <loss_sigma_3> = ', np.mean(acum))
        continue
        
        print( ' Nsignals = ', nsigs[counter].numpy(), 'TDc = ', tdc.numpy(), ' TDh = ', tdh.numpy() )
        counter+=1
        levels = 0.1*np.power(1.33, np.arange(15))
        levels = np.concatenate( [-levels[::-1], levels])

        print(' Sampling Rate        ', float(elem[2].numpy()))
        print(' Sampled Points (NC): ', tf.where( tf.abs(c[0,0,0:tdc*2:2,0])>1e-6 ).shape[0], ' of ', tdc.numpy())
        print(' Sampled Points (C):  ', tf.where( tf.abs(d[0,0,0:tdc*2:2,0])>1e-6 ).shape[0], ' of ', tdc.numpy())        
        
        plt.figure(1)
        plt.contour( tar[0,:,:], levels=levels, cmap='seismic_r' )
        plt.colorbar()
        plt.title('target ' )

        plt.figure(2)
        cc = tf.transpose(cc, perm=(0,1,3,2))
        spec_nc = tf.signal.fftshift( tf.signal.fft( cc[0,0,:,0::2].numpy() + cc[0,0,:,1::2].numpy()*1j ), axes=-1)
        plt.contour( np.transpose( spec_nc.numpy().real ), levels=levels, cmap='seismic_r' )
        plt.colorbar()
        plt.title('inp - no coupling ' )
        
        #plt.figure(3)
        #d = tf.transpose(d, perm=(0,1,3,2))
        #spec_c = np.fft.fftshift( np.fft.fft( d[0,0,:,0::2].numpy() + d[0,0,:,1::2].numpy()*1j, axis=-1), axes=-1).real
        #plt.contour( np.transpose(spec_c), levels=levels, cmap='seismic_r' )
        #plt.colorbar()
        #plt.title('inp - coupling ' )

        plt.figure(3)
        plt.contour( HCout[0,:,:], levels=levels, cmap='seismic_r' )
        plt.colorbar()
        plt.title(' pred ' )

        plt.figure(4)
        plt.contour( HCout[0,:,:] - tar[0,:,:], levels=levels/3., cmap='seismic_r' )
        plt.colorbar()
        plt.title(' pred - tar' )

        print(f' RMSD       = {np.sqrt(np.mean(np.square(HCout[0,:,:] - tar[0,:,:]))) :13.6e} ')
        print(f' Max Signal = tar: {np.max(tar) :.2f}; pred: {np.max(HCout[0,:,:]) :.2f} ')
        
        plt.show()

      sys.exit(10)  

    class SaveLoss(tf.keras.callbacks.Callback):
        def __init__(self):
            super(tf.keras.callbacks.Callback,self).__init__()

        #def on_batch_end(self,batch,logs):
        #  print('\n loss_sigma_3 ', logs['loss_sigma_3'])
        #  sys.stdout.flush()
            
        def on_epoch_end(self,epoch, logs):
            logfile=open(checkpoint_path+'/loss.out','a')
            logfile.write('%5d %10.5e %10.5e %10.5e\n' %(epoch,logs['loss'],logs['loss_sigma_3']+logs['loss_sigma_4'],logs['loss_value'] ))
            logfile.flush()
            logfile.close()
            
    history = model.fit(ds,                     \
                        epochs=EPOCHS,          \
                        verbose=1,              \
                        callbacks=[SaveLoss()]  )

    if np.isnan(history.history['loss'][0]):
      print(f' Caught a NaN is the loss')
      print(f' INFO: Restore checkpoint at {ckpt_save_path}')
      ckpt.restore(ckpt_manager.latest_checkpoint)
      continue
    
    ckpt_save_path = ckpt_manager.save()
    print(f' INFO: Saved checkpoint at {ckpt_save_path}')
    loss = history.history['loss'][0]
    lr   = model.optimizer._optimizer._decayed_lr(tf.float32).numpy()
    #lr   = model.optimizer._decayed_lr(tf.float32).numpy()

    #if history.history['loss_value'][0]<0.005:
    #  number_of_signals = number_of_signals + 1
    #  number_of_signals = np.min([number_of_signals,100])
    #if history.history['loss_value'][0]<0.0075:
    #  term0 = np.max([Sparse[0]*0.99,0.45])
    #  term1 = Sparse[1]
    #  if np.fabs(term0 - 0.45 )<1e-2:
    #     term1 = np.max([Sparse[1]*0.99,0.55])
    #  Sparse=(term0,term1)
    # 
    print(f' INFO: learning_rate = {lr :10.5e}, Signals = {number_of_signals}, SharpValue = {SharpValue :5.2f} ')
    print(f' INFO: Sparse = ({Sparse[0] :.4f},{Sparse[1] :.4f}) ')
    #print(f' INFO: SigmaScale = {model.SigmaScale.numpy() :.4f} ')

    loss_array.append(loss)
    lr_array.append(lr)
    # Write status file
    with open(StatusFile,'w') as ofs:
      ofs.write(f'number_of_signals {number_of_signals} \n')
      #ofs.write(f'SharpValue {SharpValue} \n')
      ofs.write(f'Sparse {Sparse[0]} {Sparse[1]} \n')
      #ofs.write(f'SigmaScale {model.SigmaScale.numpy() } \n')
    
    np.save( open(checkpoint_path + '/loss.npy','wb'), np.array(loss_array))
    np.save( open(checkpoint_path + '/lr.npy','wb')  , np.array(lr_array))
