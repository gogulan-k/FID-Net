import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as     tf
#import numpy as np
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

#
# Input size (batch_size, fid_length, features)
# Features include [time, real, imag]

#
# Functions to generate Poisson-Gap sampling scheme
#@tf.function
def poiss(lamby):
  L = tf.math.exp(-lamby)
  k = tf.constant(0)
  p = tf.constant([1.0])
  #
  while(p>=L):
    u = tf.random.uniform(minval=0.0, maxval=1.0, shape=(1,))
    p *= u
    k += 1
  #
  return k-tf.constant(1)

@tf.function
def poiss_dist(tot, sparsity):
  num = tf.floor( tf.cast(tot,sparsity.dtype)*sparsity)
  frac= tf.cast( tot,sparsity.dtype) / num
  num = tf.cast( num, tf.int32)
        
  adj = 2.0*(frac - 1.)

  vect = tf.zeros(shape=(tot,), dtype=tf.int32)
  n = tf.constant(0)
    
  while(n!=num):
    n = tf.constant(0)
    i = tf.constant(0)
    i1 = tf.constant(0)
    while(i<tot):

      #vect[n] = i
      vect = tf.where( tf.equal(tf.range(tot), n), i, vect )
      
      i += 1
      
      fraccy = (tf.cast(i,sparsity.dtype)+0.5)/(tf.cast(tot,sparsity.dtype) + 1.0)
      sin_fraccy = (fraccy*tf.constant(0.5*3.14159265359))
      
      k = poiss(adj*(tf.math.sin(sin_fraccy)))
      
      i+=k
      n+=1

    if (n > num):
      adj *= 1.02
    #
    elif (n < num):
      adj /= 1.02

  return vect[:num]

#@tf.function
def sched_to_idx(sampling,tot):
  idx= tf.where(tf.reshape(sampling,(1,-1)) == tf.reshape(tf.range(tot),(-1,1)),1.,0.)
  idx = tf.math.reduce_sum(idx, axis=-1)
  return idx


def LinearSharp(x, scaling=0.30):
  return scaling*x

def Sharp13C(x):
  return 0.5*x

def Sharp1H(x, SharpValue=45. ):
  return 0.8*tf.math.tanh( x / SharpValue )*SharpValue + 0.2 * x

def MakeTraining(NSignals=2,                 \
                 SW=(2000.,5000.),           \
                 NP=256,                     \
                 f1180=False,                \
                 R2=(50.,20.),              \
                 JHC=(180.,30.),             \
                 Phase=(0.,10.),             \
                 Noise=(1.2,0.12),           \
                 T1Noise=None,               \
                 Ints=(1,0.5),               \
                 NPointsIndirect=200,        \
                 AcqtIndirect=(0.030,0.050), \
                 R2Indirect=(50.,20.),       \
                 PhaseIndirect=(0.,1.),      \
                 JIndirect=(63.,10.),        \
                 BatchSize=128,              \
                 Eval=False,                 \
                 Sharp13C=False,             \
                 Sharp1H=False,              \
                 SharpFuncs=(None,None),     \
                 HScaling=(1.,),             \
                 Condense1H=False,           \
                 Sparse=None,                \
                 RandomSampling=False,       \
                 Roofing=None,               \
                 Solvent=True,              \
                 SharpValue=66.):

  RealBatchSize=BatchSize
  BatchSize=1
  #
  #
  assert isinstance( SW, (tuple, float, int))
  assert isinstance( Noise, (bool, float, tuple))
  assert isinstance( JIndirect, (tuple, float, int))
  #
  #if isinstance( NSignals, tuple):
  #  NSignals = tf.random.uniform( shape=(1,), minval=tf.reduce_min(NSignals), maxval=tf.reduce_max(NSignals), dtype=tf.dtypes.int32)
  if isinstance( SW, tuple ): 
    SW = tf.random.uniform( shape=(1,), minval=tf.reduce_min(SW), maxval=tf.reduce_max(SW))
  elif isinstance( SW, int ):
    SW = tf.constant( SW, dtype=tf.dtypes.float32)
  elif isinstance( SW, float):
    SW = tf.constant( SW, dtype=tf.dtypes.float32)
  #
  if isinstance( Noise, tuple ):
    Noise_stds = tf.random.normal( shape=(1,), mean=Noise[0], stddev=Noise[1], dtype=tf.dtypes.float32)
    Noise_stds = tf.abs(Noise_stds)
  elif isinstance( Noise, float ):
    Noise_stds = tf.constant( Noise, dtype=tf.dtypes.float32)
  elif Noise==False:
    Noise_stds=tf.constant(0.0, dtype=tf.dtypes.float32)
  else:
    sys.stderr.write(' Noise has to be <float>, <tuple>(max, min), <bool>(False) \n')
    sys.exit(10)    
  #
  # We first generate the signal in the direct dimension
  time = tf.range(NP, dtype=tf.dtypes.float32)/SW
  if f1180:
    time += tf.constant(0.5, dtype=tf.dtypes.float32)/SW
  time = time[tf.newaxis, tf.newaxis, :]  # (Batch, n_signals, time)

  pi180 = tf.math.asin(1.)/tf.constant(90.,dtype=tf.float32) # PI/180.
  twopi = tf.math.asin(1.)*tf.constant(4., dtype=tf.float32)
  mypi  = tf.math.asin(1.)*tf.constant(2., dtype=tf.float32)
  
  #css = tf.random.uniform(shape=(BatchSize,NSignals,1), minval=-0.5*0.95*SW, maxval=0.5*0.95*SW, dtype=tf.dtypes.float32)
  css = tf.random.normal(shape=(BatchSize,NSignals,1), mean=0., stddev=0.25*SW, dtype=tf.dtypes.float32)
  if Condense1H:
    css = tf.math.tanh( tf.math.pow(css/(SW*0.50),3.) )*SW*0.2 + tf.math.tanh(css/(SW*0.50))*SW*0.3
  else:
    css = tf.math.tanh(css*2./SW)*SW*0.5 # Make sure we stay within SW

  #
  if Roofing is not None:
    Roofing = tf.constant( Roofing, dtype=tf.float32)
    
  r2s = tf.abs(tf.random.normal(mean=R2[0], stddev=R2[1], shape=(BatchSize,NSignals, 1), dtype=tf.dtypes.float32))
  jchs= tf.random.normal(mean=JHC[0],  stddev=JHC[1],  shape=(BatchSize,NSignals,1), dtype=tf.dtypes.float32)
  phis= tf.random.normal(mean=Phase[0],stddev=Phase[1],shape=(BatchSize,NSignals,1), dtype=tf.dtypes.float32)*pi180
  ints= tf.random.normal(mean=Ints[0], stddev=Ints[1], shape=(BatchSize,NSignals,1), dtype=tf.dtypes.float32)
  #ints= np.ones( (BatchSize,NSignals,1))
  #
  # Make the FIDs in the directly detected proton dimension
  if Sharp1H:
    if SharpFuncs[0] is not None:
      r2s_tar = SharpFuncs[0](r2s)
    else:
      r2s_tar  = 0.8*tf.math.tanh( r2s / SharpValue )*SharpValue + 0.2 * r2s
  else:
    r2s_tar = r2s
  fids_tar = tf.complex(ints,0.) * tf.math.exp(
    tf.complex(0.,phis) +
    tf.complex( -r2s_tar*time, css*twopi*time))
  #
  # We have 5 input fids with times [-5ms, 0ms, 5ms ]
  #add_evol = tf.linspace( tf.constant(-0.005,dtype=tf.dtypes.float32),
  #                        tf.constant( 0.005,dtype=tf.dtypes.float32), 3)
  add_evol = tf.constant([0.000], dtype=tf.dtypes.float32)
  #
  # 1H-1H couplings
  j_h_1 = tf.random.normal( mean = 8., stddev=2., shape=(BatchSize,NSignals,1), dtype=tf.dtypes.float32)
  j_h_1_mask = tf.where( tf.random.uniform( shape=(BatchSize,NSignals,1), minval=0., maxval=1.0) > 0.10, 1., 0. ) #90% coupled (ortho)
  j_h_1 = j_h_1_mask * j_h_1 

  j_h_2 = tf.random.normal( mean = 4., stddev=2., shape=(BatchSize,NSignals,1), dtype=tf.dtypes.float32)
  j_h_2_mask = tf.where( tf.random.uniform( shape=(BatchSize,NSignals,1), minval=0., maxval=1.0) > 0.50, 1., 0. ) #50% coupled (meta)
  j_h_2 = j_h_2_mask * j_h_2 
  #
  #
  # We generate on a larger sw and then cut in frequency domain
  time_gen = tf.range(NP*3, dtype=tf.dtypes.float32)/(3.*SW)
  time_gen = time_gen[tf.newaxis, tf.newaxis, :]  # (Batch, n_signals, time)
  #
  #
  hmqc_coupling = tf.random.normal(mean=6.e-3, stddev=2.e-3, shape=(1,), dtype=time_gen.dtype)
  hmqc_coupling = tf.math.maximum(hmqc_coupling,0.)
  #
  # Add roofing
  cos_term1 = tf.math.cos( tf.complex(mypi*j_h_1*(time_gen+hmqc_coupling) ,0.) )
  sin_term1 = tf.math.sin( tf.complex(mypi*j_h_1*(time_gen+hmqc_coupling) ,0.) )
  #
  cos_term2 = tf.math.cos( tf.complex(mypi*j_h_2*(time_gen+hmqc_coupling) ,0.) )
  sin_term2 = tf.math.sin( tf.complex(mypi*j_h_2*(time_gen+hmqc_coupling) ,0.) )
  #
  if Roofing is not None:
    roof_scale = tf.random.normal(shape=(1,NSignals,1,2), mean=0.0, stddev=Roofing )
    #roof_scale = tf.constant( [0.25,0.25], dtype=time_gen.dtype)[tf.newaxis,tf.newaxis,tf.newaxis,:]
    comb_term1 = cos_term1 + tf.complex(0.,roof_scale[...,0]) * sin_term1
    comb_term2 = cos_term2 + tf.complex(0.,roof_scale[...,1]) * sin_term2
  else:
    comb_term1 = cos_term1
    comb_term2 = cos_term2

  #for ae in add_evol:
  fids_inp=tf.stack( [
    #tf.complex(ints,0.) * \
    #tf.complex(tf.math.cos(mypi*j_h_1*time_gen),0.) * \
    #tf.complex(tf.math.cos(mypi*j_h_2*time_gen),0.) * \
    #tf.math.exp( tf.complex(0., phis)  \
    #             + tf.complex( 0., css*2.*mypi*(time_gen + add_evol[0] )) \
    #             - tf.complex(r2s*time_gen,0.)),
    #tf.complex(ints,0.) * \
    #tf.complex(tf.math.cos(mypi*j_h_1*time_gen),0.) * \
    #tf.complex(tf.math.cos(mypi*j_h_2*time_gen),0.) * \
    #tf.math.exp( tf.complex(0., phis)  \
    #             + tf.complex( 0., css*2.*mypi*(time_gen + add_evol[1] )) \
    #             - tf.complex(r2s*time_gen,0.)),
    tf.complex(ints,0.) * \
    comb_term1 * comb_term2 * \
    tf.math.exp( tf.complex(0., phis)  \
                 + tf.complex( 0., css*2.*mypi*(time_gen + add_evol[0] )) \
                 - tf.complex(r2s*time_gen,0.))], axis=1)
                     
  #fids_inp = tf.stack( fids_inp, axis=1)
  #
  #window = tf.ones( shape(3*NP,), dtype=tf.float32)
  #window[0] *= 0.5
  window = tf.where( tf.range(3*NP, dtype=tf.float32) > 0., 1., 0.5)
  window = tf.complex(tf.reshape( window, (1,1,1,-1)),0.)

  fids_inp_ft = tf.signal.fftshift( tf.signal.fft(fids_inp * window ), axes=-1)
  #
  # Make water baselines
  h2o_ints = 10.*tf.random.normal(mean=Ints[0], stddev=Ints[1], shape=(1,NPointsIndirect,4,1), dtype=tf.float32)
  sign_mask = tf.where( tf.random.uniform( shape=(1,NPointsIndirect,4,1), minval=0., maxval=1.0 ) >0.5, 1., -1.)
  h2o_ints = sign_mask * h2o_ints
  slp_factor= tf.random.uniform( shape=(1,NPointsIndirect,4,1), minval=0., maxval=1.0, dtype=tf.float32)
  h2o_slopes = h2o_ints*(slp_factor*(1./( tf.linspace(tf.constant(0.,dtype=tf.float32),tf.constant(1.,dtype=tf.float32),2*NP) + 0.1 ) - 0.9) - tf.linspace(tf.constant(0.,dtype=tf.float32),tf.constant(1.,dtype=tf.float32),2*NP)*(1.-slp_factor))
  sign_mask = tf.where( tf.random.uniform( shape=(1,NPointsIndirect,4,1), minval=0., maxval=1.0 ) >0.5, 1., -1.)
  h2o_slopes *= sign_mask

  left_right = tf.random.uniform(shape=(1,), minval=0., maxval=1.0, dtype=tf.float32)

  h2o_slopes_nc = tf.complex( h2o_slopes[:,:,0,:], h2o_slopes[:,:,1,:] )
  h2o_slopes_nc = tf.expand_dims( h2o_slopes_nc, axis=0)
  h2o_slopes_nc = tf.where(  left_right > 0.5, h2o_slopes_nc[:,:,:,::-1], h2o_slopes_nc )
  #h2o_slopes_nc = tf.signal.ifft( tf.signal.fftshift(h2o_slopes_nc, axes=-1))

  h2o_slopes_c = tf.complex( h2o_slopes[:,:,2,:], h2o_slopes[:,:,3,:] )
  h2o_slopes_c = tf.expand_dims( h2o_slopes_c, axis=0)
  h2o_slopes_c = tf.where(  left_right > 0.5, h2o_slopes_c[:,:,:,::-1], h2o_slopes_c )
  #h2o_slopes_c = tf.signal.ifft( tf.signal.fftshift(h2o_slopes_c, axes=-1))

  fids_inp = 0.3333*tf.signal.ifft( tf.signal.fftshift(fids_inp_ft[:,:,:,NP:2*NP], axes=-1) )/window[0,0,0,:NP]
  
  #
  # Make the signals in the directly indirect dimension
  acqt_in = tf.random.uniform( shape=(1,), minval=tf.reduce_min(AcqtIndirect), maxval=tf.reduce_max(AcqtIndirect), dtype=tf.float32)
  np_in   = tf.constant(NPointsIndirect, dtype=tf.dtypes.int32)
  sw_in   = tf.cast(np_in,tf.float32) / acqt_in
  times_in= tf.range(np_in, dtype=tf.float32)/sw_in

  css_in = tf.random.uniform( shape=(BatchSize,NSignals,1), minval=-0.5*sw_in, maxval=0.5*sw_in, dtype=tf.float32)

  r2s_in = tf.abs( tf.random.normal(mean=R2Indirect[0], stddev=R2Indirect[1], shape=(BatchSize,NSignals,1), dtype=tf.float32))
  phis_in= tf.random.normal(mean=PhaseIndirect[0], stddev=PhaseIndirect[1], shape=(BatchSize,NSignals,1), dtype=tf.float32)*pi180
  if Sharp13C:
    if SharpFuncs[1] is not None:
      r2s_in_tar = SharpFuncs[1](r2s_in)
    else:
      r2s_in_tar  = 0.8*tf.math.tanh( r2s_in / SharpValue )*SharpValue + 0.2 * r2s_in
  else:
    r2s_in_tar = r2s_in

  #
  # Add coupling
  #print( 'SW indirect ', sw_in )
  #print( 'Acq_in      ', np.max(times_in))
  #print( 'np_in       ', np_in )
  if isinstance( JIndirect, tuple):
    j_in_1 = tf.random.normal( mean = JIndirect[0], stddev=JIndirect[1], shape=(BatchSize,NSignals,1), dtype=tf.float32)
    j_in_2 = tf.random.normal( mean = JIndirect[0], stddev=JIndirect[1], shape=(BatchSize,NSignals,1), dtype=tf.float32)
  elif isinstance( JIndirect, float ):
    j_in_1 = tf.ones( (BatchSize, NSignals,1))*tf.constant(JIndirect, dtype=tf.float32)
    j_in_2 = tf.ones( (BatchSize, NSignals,1))*tf.constant(JIndirect, dtype=tf.float32)
  else:
    j_in_1 = tf.zeros( shape=(BatchSize,NSignals,1), dtype=tf.float32)
    j_in_2 = tf.zeros( shape=(BatchSize,NSignals,1), dtype=tf.float32)
  #
  j_in_1_mask = tf.where( tf.random.uniform( shape=(BatchSize,NSignals,1), minval=0., maxval=1., dtype=tf.float32) > 0.8, 0., 1. )
  j_in_1 = j_in_1_mask * j_in_1
  #
  j_in_2_mask = tf.where( tf.random.uniform( shape=(BatchSize,NSignals,1), minval=0., maxval=1., dtype=tf.float32) > 0.8, 0., 1. )
  j_in_2 = j_in_2_mask * j_in_2

  #
  # Full indirect coupling evolution
  #
  # No coupling evolution
  if T1Noise is not None:
    t1noise = tf.random.normal(shape=times_in.shape, mean=1.0, stddev=T1Noise, dtype=times_in.dtype)
  else:
    t1noise = tf.ones(shape=times_in.shape, dtype=times_in.dtype)

  #
  # Take into account coupling evolution during pulses
  pulse_delay = tf.random.normal(shape=(1,), mean=100e-6, stddev=30e-6, dtype=times_in.dtype)
  pulse_delay = tf.math.maximum(pulse_delay,0.)

  cos_term1 = tf.math.cos( tf.complex(mypi*j_in_1*(times_in +pulse_delay ),0.) )
  sin_term1 = tf.math.sin( tf.complex(mypi*j_in_1*(times_in +pulse_delay ),0.) )
  #
  cos_term2 = tf.math.cos( tf.complex(mypi*j_in_2*(times_in +pulse_delay ),0.) )
  sin_term2 = tf.math.sin( tf.complex(mypi*j_in_2*(times_in +pulse_delay ),0.) )
  #
  if Roofing is not None:
    roof_scale = tf.random.normal(shape=(1,NSignals,1,2), mean=0.0, stddev=Roofing )
    comb_term1 = cos_term1 + tf.complex(0.,roof_scale[...,0]) * sin_term1
    comb_term2 = cos_term2 + tf.complex(0.,roof_scale[...,1]) * sin_term2
  else:
    comb_term1 = cos_term1
    comb_term2 = cos_term2
    
  fid_in_nc = comb_term1 * comb_term2 * \
              tf.exp( tf.complex(0.,phis_in) + \
                      tf.complex(0.,css_in*2.*mypi*times_in * t1noise) - \
                      tf.complex(r2s_in*(times_in), 0.))  # (Batch, n_signals, time )
  #
  # Coupling evolution
  tau_coup =  tf.constant(0.0023, dtype=tf.float32)
  #
  if T1Noise is not None:
    t1noise = tf.random.normal(shape=times_in.shape, mean=1.0, stddev=T1Noise, dtype=times_in.dtype)

  cos_term1 = tf.math.cos( tf.complex(mypi*j_in_1*(times_in + tau_coup + pulse_delay ),0.) )
  sin_term1 = tf.math.sin( tf.complex(mypi*j_in_1*(times_in + tau_coup + pulse_delay ),0.) )
  #
  cos_term2 = tf.math.cos( tf.complex(mypi*j_in_2*(times_in + tau_coup + pulse_delay ),0.) )
  sin_term2 = tf.math.sin( tf.complex(mypi*j_in_2*(times_in + tau_coup + pulse_delay ),0.) )

  if Roofing is not None:
    comb_term1 = cos_term1 + tf.complex(0.,roof_scale[...,0]) * sin_term1
    comb_term2 = cos_term2 + tf.complex(0.,roof_scale[...,1]) * sin_term2
  else:
    comb_term1 = cos_term1
    comb_term2 = cos_term2  
    
  fid_in_c =  comb_term1 * comb_term2 * \
              tf.exp( tf.complex(0.,phis_in) + \
                      tf.complex(0., css_in*2.*mypi*times_in *t1noise) - \
                      tf.complex(r2s_in*(times_in+tau_coup),0.) )  # (Batch, n_signals, time )  
  #
  # Target
  fid_in_tar= tf.exp( tf.complex(0., phis_in) + \
                      tf.complex(0.,css_in*2.*mypi*times_in) - \
                      tf.complex(r2s_in_tar*times_in,0.) )  # (Batch, n_signals, time )
  #
  pts_to_use = tf.math.floor(tf.random.uniform(shape=(BatchSize,), minval=96, maxval=NPointsIndirect, dtype=tf.dtypes.float32))
  pts_to_use = tf.expand_dims(pts_to_use, axis=-1)
  pts_mask =   tf.where( tf.tile( tf.expand_dims( tf.range(200, dtype=tf.dtypes.float32), axis=0 ), [BatchSize,1]) < pts_to_use, 1., 0.)
  pts_mask =   tf.expand_dims(pts_mask, axis=0) # (1,pts_mask)
  #
  # Make sparse sampling
  sr=tf.constant([1.,], dtype=tf.float32)
  if Sparse is not None:

    assert isinstance( Sparse, (tuple, float))
    
    if isinstance( Sparse, tuple ): 
      sr = tf.random.uniform( shape=(1,), minval=Sparse[0], maxval=Sparse[1], dtype=tf.float32)
    elif isinstance( Sparse, float):
      sr = tf.constant( [Sparse,], dtype=tf.float32 )
    else:
      print(' Internal error w.r.t. to Sparse ', file=sys.stderr)
      sys.stderr.flush()

    if RandomSampling:
      ss_mask = tf.where( tf.random.uniform( shape=(BatchSize,2*(NPointsIndirect-1)), minval=0., maxval=1.0) < sr, 1., 0. )

      ss_mask = tf.concat( [tf.constant([[1.,]], dtype=ss_mask.dtype),
                            ss_mask[...,0::2],
                            tf.constant([[1.,]], dtype=ss_mask.dtype),
                            ss_mask[...,1::2] ], axis=-1)
      
      ss_mask_nc = ss_mask[...,:NPointsIndirect]
      ss_mask_c  = ss_mask[...,NPointsIndirect:]

    else:

      sampling_nc = poiss_dist(\
                               tf.cast(pts_to_use[0,0],tf.int32),  \
                               sr[0]        \
      )
      ss_mask_nc = tf.reshape( sched_to_idx(sampling_nc,tf.constant(NPointsIndirect)), (1,-1))

      sampling_c = poiss_dist(\
                              tf.cast(pts_to_use[0,0],tf.int32),  \
                              sr[0]        \
      )
      ss_mask_c = tf.reshape( sched_to_idx(sampling_c,tf.constant(NPointsIndirect)), (1,-1))
    
    pts_mask_nc = pts_mask #* ss_mask_nc 
    pts_mask_c  = pts_mask #* ss_mask_c
  else:
    pts_mask_nc = pts_mask
    pts_mask_c  = pts_mask
  #
  # We need to mask the fids
  fid_in_nc = tf.complex(pts_mask ,0.) * fid_in_nc
  fid_in_c  = tf.complex(pts_mask ,0.) * fid_in_c
  fid_in_tar= tf.complex(pts_mask,0.) * fid_in_tar
  times_in  = pts_mask * times_in 
  #
  # (this window is for the target only)
  window_in = tf.where( tf.range(2*NPointsIndirect, dtype=tf.float32) > 0., 1., 0.5)
  window_in = tf.reshape( window_in, (1,1,-1))
  window_in = tf.complex(window_in, 0.)
  #
  # Zero fill
  fid_in_tar= tf.pad( fid_in_tar,[[0,0],[0,0],[0,2*NPointsIndirect-fid_in_tar.shape[-1]]], constant_values=0.)
  #
  spec_in_tar= tf.math.real(tf.signal.fftshift( tf.signal.fft( fid_in_tar * window_in ), axes=-1 ))  
  #
  # Joind shapes together
  spec_in_tar = tf.expand_dims( spec_in_tar, axis=-1)   #(Batch, NSignals, IndirectNP*2, 1 )

  pts_to_use_H = tf.math.scalar_mul(tf.constant(1, dtype=tf.dtypes.float32),
                                    tf.math.floor(tf.random.uniform(shape=(1,), minval=NP/2, maxval=NP, dtype=tf.dtypes.float32)))
  pts_to_use_H = tf.expand_dims(pts_to_use_H, axis=-1)
  pts_mask_h   = tf.where( tf.expand_dims( tf.range(NP, dtype=tf.dtypes.float32), axis=0 ) < pts_to_use_H, 1., 0.) #(1,512)
  pts_mask_h_1 = tf.cast(tf.expand_dims(pts_mask_h, axis=0), tf.float32) # (1,1,512) 
  pts_mask_h_2 = tf.expand_dims(pts_mask_h_1, axis=0) # (1,1,1,512)

  fids_tar = fids_tar * tf.complex(pts_mask_h_1, 0.)
  fids_inp = fids_inp * tf.complex(pts_mask_h_2, 0.)
  time = time * pts_mask_h_1 

  h2o_slopes_nc = tf.signal.ifft( tf.signal.fftshift(h2o_slopes_nc, axes=-1))[:,:,:,:NP] # (1,1,200,np)
  h2o_slopes_nc *= tf.complex(pts_mask_h_2, 0.)
  h2o_slopes_nc = tf.pad( h2o_slopes_nc, [ [0,0],[0,0],[0,0],[0,h2o_slopes_nc.shape[-1] ]], "CONSTANT", constant_values=0.)
  h2o_slopes_nc = tf.signal.fftshift( tf.signal.fft( h2o_slopes_nc ), axes=-1)

  h2o_slopes_c = tf.signal.ifft( tf.signal.fftshift(h2o_slopes_c, axes=-1))[:,:,:,:NP] # (1,1,200,np)
  h2o_slopes_c *= tf.complex(pts_mask_h_2, 0.)
  h2o_slopes_c = tf.pad( h2o_slopes_c, [ [0,0],[0,0],[0,0],[0,h2o_slopes_c.shape[-1] ]], "CONSTANT", constant_values=0.)
  h2o_slopes_c = tf.signal.fftshift( tf.signal.fft( h2o_slopes_c ), axes=-1)

  fids_tar = tf.expand_dims( fids_tar, axis=-2) #(batch, NSignals, 1, NP)
  fids_inp = tf.expand_dims( fids_inp, axis=-2) #(batch, coup, NSignals, 1, NP )

  #
  # Zero fill
  spec_tar= tf.pad( fids_tar,[[0,0],[0,0],[0,0],[0,fids_tar.shape[-1]]],"CONSTANT", constant_values=0.)  
  spec_inp= tf.pad( fids_inp,[[0,0],[0,0],[0,0],[0,0],[0,fids_tar.shape[-1]]],"CONSTANT", constant_values=0.)  
  
  windowH = tf.concat( [ tf.constant([0.5], dtype=tf.dtypes.float32), tf.ones( spec_tar.shape[-1]-1, dtype=tf.dtypes.float32)], axis=0)
  windowH = tf.expand_dims( windowH, axis=0)
  windowH = tf.complex( windowH, 0.)

  spec_tar = tf.signal.fftshift( tf.signal.fft( windowH * spec_tar ) ,axes=-1)
  spec_tar = tf.math.real(spec_tar)

  spec_inp = tf.signal.fftshift( tf.signal.fft( windowH * spec_inp ) ,axes=-1)
  spec_inp = tf.math.real(spec_inp)

  #
  # Take direct product
  fid_in_nc = tf.expand_dims( tf.expand_dims( fid_in_nc, axis=-1 ), axis=1)
  fid_in_c  = tf.expand_dims( tf.expand_dims( fid_in_c,  axis=-1 ), axis=1)
  
  fids_2d_tar    = spec_in_tar @ spec_tar  # (1, NSignals, np_in, np)        # ok
  fids_2d_inp_nc = fid_in_nc @ tf.complex( spec_inp, 0.) # (1, coup, NSignals, np_in:time, np:freq)
  fids_2d_inp_c  = fid_in_c  @ tf.complex( spec_inp, 0.) # ()
  
  #
  # add noise
  fid_noise_nc = tf.complex( \
                             tf.random.normal(mean=0., stddev=Noise_stds, shape=(BatchSize,1,fids_2d_inp_nc.shape[3],2*NP), dtype=tf.float32), \
                             tf.random.normal(mean=0., stddev=Noise_stds, shape=(BatchSize,1,fids_2d_inp_nc.shape[3],2*NP), dtype=tf.float32) )

  fid_noise_c = tf.complex( \
                             tf.random.normal(mean=0., stddev=Noise_stds, shape=(BatchSize,1,fids_2d_inp_nc.shape[3],2*NP), dtype=tf.float32), \
                             tf.random.normal(mean=0., stddev=Noise_stds, shape=(BatchSize,1,fids_2d_inp_nc.shape[3],2*NP), dtype=tf.float32) )

  fids_2d_tar    = tf.reduce_sum( fids_2d_tar, axis=1)                 # (1,np_in, np)
  fids_2d_inp_nc = tf.reduce_sum( fids_2d_inp_nc, axis=2) + fid_noise_nc  # (1,coup, np_in, np )
  fids_2d_inp_c  = tf.reduce_sum( fids_2d_inp_c, axis=2) + fid_noise_c  # (1,coup, np_in, np )  

  if Solvent:
    fids_2d_inp_nc += h2o_slopes_nc * tf.complex( tf.reshape( pts_mask, (1,1,-1,1)), 0. ) 
    fids_2d_inp_c  += h2o_slopes_c   * tf.complex( tf.reshape( pts_mask, (1,1,-1,1)), 0. ) 
  #
  # Do Sparse sampling
  if Sparse is not None:
    ss_mask_nc_rs = tf.reshape( ss_mask_nc, (1,1,-1,1))
    ss_mask_c_rs  = tf.reshape( ss_mask_c , (1,1,-1,1))
    fids_2d_inp_nc = tf.complex(ss_mask_nc_rs,0.) * fids_2d_inp_nc
    fids_2d_inp_c  = tf.complex(ss_mask_c_rs ,0.) * fids_2d_inp_c    
  #
  # Convert to tensorflow Tensors
  time = tf.tile(time[:,0,:],(BatchSize,1) )
  time = tf.expand_dims(time, axis=1)
  
  time = time + tf.reshape( add_evol, (1,-1,1)) #(batch,coup,np)
  time= tf.reshape(tf.stack([time,time],             axis=-1),(BatchSize,1,2*NP))  # (batch,coup,(time.r,time.i))
  if Sparse is not None:
    times_in= tf.reshape(tf.stack([times_in *ss_mask_nc,times_in*ss_mask_c], axis=-1),(BatchSize,2*np_in))  #(Batch, (time.r, time.i ))
  else:
    times_in= tf.reshape(tf.stack([times_in,times_in], axis=-1),(BatchSize,2*np_in))  #(Batch, (time.r, time.i ))    
  #
  # Order points
  fids_2d_inp_nc = tf.transpose( fids_2d_inp_nc, perm=(0,1,3,2))
  fids_2d_inp_nc = tf.reshape(tf.stack([tf.math.real(fids_2d_inp_nc),tf.math.imag(fids_2d_inp_nc)],axis=-1),(BatchSize,1,2*NP,-1))
  fids_2d_inp_nc = tf.transpose( fids_2d_inp_nc, perm=(0,1,3,2))

  fids_2d_inp_c = tf.transpose( fids_2d_inp_c, perm=(0,1,3,2))
  fids_2d_inp_c = tf.reshape(tf.stack([tf.math.real(fids_2d_inp_c),tf.math.imag(fids_2d_inp_c)],axis=-1),(BatchSize,1,2*NP,-1))
  fids_2d_inp_c = tf.transpose( fids_2d_inp_c, perm=(0,1,3,2))
  #
  # Normalise each batch
  norm_factor = tf.reduce_max( tf.abs(fids_2d_inp_nc[:,:,:,:]), axis=(1,2,3))
  #
  fids_2d_inp_nc= fids_2d_inp_nc/norm_factor
  fids_2d_inp_c = fids_2d_inp_c/norm_factor
  fids_2d_tar   = fids_2d_tar/norm_factor
  #
  if not Eval:
    time    = tf.cast(time, tf.float32)
    times_in = tf.cast(times_in, tf.float32)
    fids_2d_inp_nc= tf.cast(fids_2d_inp_nc,tf.float32)
    fids_2d_inp_c = tf.cast(fids_2d_inp_c, tf.float32)
    fids_2d_tar   = tf.cast(fids_2d_tar,   tf.float32)

  time = tf.squeeze( time, axis=0)
  times_in = tf.squeeze( times_in, axis=0)
  fids_2d_inp_nc = tf.squeeze( fids_2d_inp_nc, axis=0)
  fids_2d_inp_c  = tf.squeeze( fids_2d_inp_c,  axis=0)
  fids_2d_tar    = tf.squeeze( fids_2d_tar, axis=0)

  if Eval:
    return ((time, times_in, fids_2d_inp_nc, fids_2d_inp_c), fids_2d_tar, sr )
  else:
    return ((time, times_in, fids_2d_inp_nc, fids_2d_inp_c), fids_2d_tar )    


