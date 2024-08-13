#!/usr/bin/env python3

"""
  Script to process aromatic 13C-1H correlation maps.
  Spectra should be recorded with the dedicated pulse programme

  Written by D Flemming Hansen
  April 2024, flemming.hansen@crick.ac.uk / d.hansen@ucl.ac.uk

  Citation: VK Shukla, G Karunanithy, P Vallurupalli, DF Hansen
            bioRxiv (2024),https://doi.org/10.1101/2024.04.01.587635 
"""

from fidnet.aromatic_fidnet2.hansenlab import MYSTRING

counter=0
for l in MYSTRING:
  if counter>3 and counter<17:
    print(l[258+133+131:])
  counter+=1

  
import tensorflow as tf
import numpy      as np 
import nmrglue    as ng
import logging
from pathlib import Path
from fidnet.aromatic_fidnet2.FIDNet import *

#
# Let us check the compute power of the GPU (if there)

def check_gpus(UseGPU, GPUIDX):
  gpus=tf.config.list_physical_devices('GPU')

  if len(gpus)==0:
    print(f' INFO: Found {len(gpus)} GPUs installed. Please check that ')
    print(f'       you have a CUDA version > 11.2 installed ')
  else:
    print(f' INFO: Found {len(gpus)} GPUs installed')  
    
  if UseGPU and len(gpus)>0:
    #
    if GPUIDX is None:
      #Chose the fastest GPU
      #
      compute_capability=[]
      name=[]

      try:
        for g in gpus:
          compute_capability.append(tf.config.experimental.get_device_details(g)['compute_capability'][0])
          name.append(tf.config.experimental.get_device_details(g)['device_name'])

          GPUIDX = np.argmax(compute_capability)
      except(AttributeError):      
        GPUIDX=0
      
      try:
        GPUNAME=tf.config.experimental.get_device_details(gpus[GPUIDX])['device_name']
      except(AttributeError):
        GPUNAME=None

    if GPUNAME is not None:
      print(f' INFO: Processing on GPU:/{GPUIDX} with name {GPUNAME}')
    else:
      print(f' INFO: Processing on GPU:/{GPUIDX}')
      
    strategy = tf.distribute.OneDeviceStrategy(device='/gpu:%d' %(GPUIDX,))
    
  else:
    print(f' INFO: Processing on CPU ')  
    strategy = tf.distribute.OneDeviceStrategy(device='/cpu')
  
  return strategy
  
#
#Suppress tensorflow warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def _aromatic_fidnet2(infile, outfile, model_weights, UseGPU, GPUIDX, offset1h, offset13c):
  strategy = check_gpus(UseGPU, GPUIDX)
  with strategy.scope():
    
    fidnet_2d_model = CombinedFIDNet(
      fidnet_filters_1h=20,     \
      fidnet_filters_13c=20,    \
      blocks=3,                 \
      fidnet_kernel=(7,10),     \
      refine_kernel=(13,13),    \
      rate=0.10 )

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
        self.SigmaScale = tf.Variable( initial_value= tf.constant(0.0, dtype=tf.float32), name='SigmaScale', trainable=False)


      def train_step(self,data):
        return {m.name: m.result() for m in self.metrics}
      
      def get_config(self):
        return {}

    inputs = (tf.keras.Input(shape=(1,512)),     \
              tf.keras.Input(shape=(400,)),      \
              tf.keras.Input(shape=(1,400,512)), \
              tf.keras.Input(shape=(1,400,512))  )
    outputs = fidnet_2d_model([inputs[0], inputs[1], inputs[2], inputs[3] ])
    model = MyModel(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse') 

    
    model.load_weights(model_weights)
    
  infile_nc  = infile
  if outfile is None:
    outfile = infile.parent / Path('dec.ft2')
  else:
    outfile = Path(outfile).with_suffix('.ft2')

  esdfile = outfile.parent / 'esd_aromatic.ft2'
  inpfile = outfile.parent / 'inp_aromatic.ft2'

  dic_nc, data_nc = ng.pipe.read(infile_nc)

  data_c = data_nc[1,:,:]
  data_nc= data_nc[0,:,:]
  data_c = np.expand_dims( data_c, axis=0 )
  data_nc= np.expand_dims( data_nc, axis=0 )
  
  data_orig_nc = np.copy(data_nc)
  data_orig_c  = np.copy(data_c)
  SWh = float(dic_nc['FDF2SW'])
  SWc = float(dic_nc['FDF3SW'])


  npts_c = data_nc.shape[-2]
  npts_h = int(dic_nc['FDF2APOD'])
  npts_c = int(dic_nc['FDF3TDSIZE'])  # This is complex 

  print(' INFO: Start processing: ')
  print(f'    1H  points used : {npts_h}' )
  print(f'    13C points used : {npts_c}' )
  print(f'    Use sine-square window function with offset={offset1h} in 1H')
  print(f'    Use sine-square window function with offset={offset13c} in 13C')

  if npts_h < 128 or npts_h>256:
    print(f' !WARNING!: The FID-Net-2 model is only trained with number of ')
    print(f'            1H complex points between 128 and 256')
  if npts_c < 96 or npts_c>200:
    print(f' !WARNING!: The FID-Net-2 model is only trained with number of ')
    print(f'            13C complex points between 96 and 200')

  #
  # Define the proton dimension
  times  = tf.constant( np.arange(npts_h)/SWh, dtype=tf.dtypes.float32)
  times  = tf.reshape(tf.stack([times, times], axis=-1), (2*npts_h,))
  times  = tf.reshape( times, (1,-1))

  times_in = tf.constant( np.arange(npts_c)/SWc, dtype=tf.dtypes.float32)
  times_in = tf.reshape(tf.stack([times_in, times_in], axis=-1), (2*npts_c,))

  #
  # Normalise
  norm_factor = np.max( np.abs(data_nc) )
  #
  data_nc = data_nc/norm_factor
  data_c  = data_c/norm_factor

  #
  # pad to the correct length
  fid_nc  = tf.pad( data_nc,[[0,0],[0,400 - data_nc.shape[-2] ],[0,0]],"CONSTANT", constant_values=0.)
  fid_c   = tf.pad( data_c ,[[0,0],[0,400 - data_c.shape[-2]],[0,0] ], "CONSTANT", constant_values=0.)
  times   = tf.pad( times,  [[0,0],[0,512-times.shape[-1]]],        "CONSTANT", constant_values=0.)
  times_in= tf.pad( times_in, [ [0,400-times_in.shape[-1]]],        "CONSTANT", constant_values=0.)

  times    = tf.expand_dims( times, axis=0)
  times_in = tf.expand_dims( times_in, axis=0)
  fid_c    = tf.expand_dims( fid_c, axis=0)
  fid_nc   = tf.expand_dims( fid_nc, axis=0)

  #
  # Standard Fourier Transform of input spectrum
  window = tf.concat( [tf.constant([0.5,], dtype=tf.float32), tf.ones( shape=(fid_nc.shape[2]//2-1,), dtype=tf.float32)], axis=-1 )
  window = tf.reshape( window, (1,1,1,-1))
  window= tf.complex( window, 0.)

  spec_nc = window * tf.transpose( tf.complex(fid_nc[:,:,0::2,:],fid_nc[:,:,1::2,:]),perm=(0,1,3,2))
  spec_nc = tf.pad( spec_nc, [ [0,0],[0,0],[0,0],[0, spec_nc.shape[-1]]], constant_values=0.)
  spec_nc = tf.signal.fftshift( tf.signal.fft( spec_nc ), axes=-1)

  spec_c  = window * tf.transpose( tf.complex(fid_c[:,:,0::2,:], fid_c[:,:,1::2,:]), perm=(0,1,3,2))
  spec_c  = tf.pad( spec_c, [ [0,0],[0,0],[0,0],[0, spec_c.shape[-1]]], constant_values=0.)
  spec_c  = tf.signal.fftshift( tf.signal.fft( spec_c ), axes=-1)


  HCout = model.predict([times, times_in, fid_nc, fid_c], verbose=0)
  HCout, Esd = HCout[...,0], HCout[...,1]

  #
  # We should chop a few points in both 1H and 13C
  if True:
    npts_h = int(dic_nc['FDF2APOD']) - 8   # 8 Real points
    npts_c = int(dic_nc['FDF3TDSIZE']) - 4 # 4 This is complex 

    times  = tf.constant( np.arange(npts_h)/SWh, dtype=tf.dtypes.float32)
    times  = tf.reshape(tf.stack([times, times], axis=-1), (2*npts_h,))
    times  = tf.reshape( times, (1,-1))
    
    times_in = tf.constant( np.arange(npts_c)/SWc, dtype=tf.dtypes.float32)
    times_in = tf.reshape(tf.stack([times_in, times_in], axis=-1), (2*npts_c,))
    
    times   = tf.pad( times,  [[0,0],[0,512-times.shape[-1]]],        "CONSTANT", constant_values=0.)
    times_in= tf.pad( times_in, [ [0,400-times_in.shape[-1]]],        "CONSTANT", constant_values=0.)

    times    = tf.expand_dims( times, axis=0)
    times_in = tf.expand_dims( times_in, axis=0)


  HCout , _ = convolve(HCout, times_in, transpose=True, offset=offset13c)
  HCout , _ = convolve(HCout, times, transpose=False, offset=offset1h)

  Esd , _ = convolve(Esd, times_in, transpose=True, offset=offset13c)
  Esd , _ = convolve(Esd, times, transpose=False, offset=offset1h)

  spec_nc = tf.squeeze( spec_nc, axis=0)
  spec_nc = tf.math.real( tf.transpose( spec_nc, perm=(0,2,1)) )
  spec_nc , _ = convolve(spec_nc, times_in, transpose=True, offset=offset13c)
  spec_nc , _ = convolve(spec_nc, times, transpose=False, offset=offset1h)

  sigma = tf.math.scalar_mul(0.998, tf.math.sigmoid(Esd)) + tf.constant( 0.001, dtype=HCout.dtype)
  sigma = tf.math.reciprocal_no_nan(sigma)
  sigma = tf.math.subtract( sigma, tf.constant(1., dtype=sigma.dtype))
  sigma = tf.math.scalar_mul(0.5,sigma)

  HCout = tf.squeeze( HCout, axis=0)[::-1,:]
  HCout *= norm_factor

  sigma = tf.squeeze( sigma, axis=0)[::-1,:]
  sigma *= norm_factor

  spec_nc = tf.squeeze( spec_nc, axis=0)[::-1,:]
  spec_nc *= norm_factor

  #
  # Eliminate coupling dimension
  dic_nc['FDF1TDSIZE']=1.0
  dic_nc['FDF1APOD']=1.0
  dic_nc['FDF3SIZE']=1.0
  dic_nc['FDDIMCOUNT']=2.0
  #
  # Adjust for FFT of 1H
  dic_nc['FDF2QUADFLAG']=1.0
  dic_nc['FDF2TDSIZE']=512.0
  dic_nc['FDSIZE']=512.0
  dic_nc['FDF2FTFLAG']=1.0
  dic_nc['FDF2FTSIZE']=512
  dic_nc['FDQUADFLAG']=1.0
  #
  # Adjust for 13C
  dic_nc['FDF3SIZE'] = 1.
  dic_nc['FDF3FTFLAG']=1.0
  dic_nc['FDF3QUADFLAG']=1.0
  dic_nc['FDSPECNUM'] = 400.
  dic_nc['FDSLICECOUNT'] = 400.


  print(' INFO: Spectra saved:')
  print(f'    Spectrum ({outfile}) and uncertainties ({esdfile}) ')
  print(f'    Input spectrum for reference ({inpfile}) ')
  print('    saved with shape', HCout.shape, 'in nmrPipe format')

  ng.pipe.write(str(outfile),dic_nc, HCout.numpy().real, overwrite=True)
  ng.pipe.write(str(esdfile),dic_nc, sigma.numpy().real, overwrite=True)
  ng.pipe.write(str(inpfile),dic_nc, spec_nc.numpy().real, overwrite=True)

