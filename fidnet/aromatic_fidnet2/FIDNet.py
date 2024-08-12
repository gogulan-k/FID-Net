"""

  This is the FID-Net-2 DNN archetecture.
  If you use this, please cite:

  Shukla, Karunanithy, Vallurupalli, Hansen
  bioRxiv (2024),
  https://doi.org/10.1101/2024.04.01.587635  

  Flemming, 2022-2024, flemming.hansen@crick.ac.uk / d.hansen@ucl.ac.uk

"""

import os,sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as     tf
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

#
# define my own activations to help the gradient optimizer
def dfh_tanh(x):
    return tf.math.tanh(x)*tf.constant(0.98, dtype=x.dtype) + x*tf.constant(0.02,dtype=x.dtype)
def dfh_sigmoid(x):
    return tf.math.sigmoid(x)*tf.constant(0.98, dtype=x.dtype) + x*tf.constant(0.02,dtype=x.dtype)
def dfh_relu(x):
    return tf.nn.relu(x)*tf.constant(0.98, dtype=x.dtype) + x*tf.constant(0.02,dtype=x.dtype)

def hilbert(x):
    """
    Compute the analytic signal, using the Hilbert transform in tensorflow.
    The transformation is done along the last axis by default.

    Flemming, March 2022

    """
    if x.dtype.is_complex:
        raise ValueError("x must be real.")
    #
    N = x.get_shape()[-1]
    #
    # Do forward fft
    Xf = tf.signal.fft(tf.cast(x,dtype=tf.complex64))
    #
    # Make unit-step function vector 
    hh = tf.concat( [tf.constant([1.], dtype=x.dtype),
                     2*tf.ones(N//2-1, dtype=x.dtype),
                     tf.constant([1.], dtype=x.dtype),
                     tf.zeros(N//2-1,  dtype=x.dtype),
                    ], axis=-1)
    hh = tf.complex( hh, 0. )
    X_conv = tf.math.multiply(Xf,hh)
    #
    # inverse fft
    X_ifft = tf.signal.ifft(X_conv)
    return 1.0*X_ifft


class FIDNetLayer(tf.keras.layers.Layer):
  def __init__(self, filters=32, kernel=(3,8), blocks=1, dilations=[1,2,3,4,6,8,10,12,14,16,18,20,24,28,32]):
    super(FIDNetLayer, self).__init__()
    self.filters=filters
    self.kernel=kernel
    self.blocks=blocks
    self.dilations=dilations

    self.conv_y1=[]
    self.conv_y2=[]
    self.conv_z0=[]
    self.dense_z=[]

    #define layers
    for b in range(self.blocks):
        for i in range(len(self.dilations)):
            dil = self.dilations[i]
            self.conv_y1.append( tf.keras.layers.Conv2D(filters=self.filters,   kernel_size=self.kernel, padding='valid', dilation_rate=[1,dil] ))
            self.conv_y2.append( tf.keras.layers.Conv2D(filters=self.filters,   kernel_size=self.kernel, padding='valid', dilation_rate=[1,dil] ))
            self.conv_z0.append( tf.keras.layers.Conv2D(filters=self.filters*2, kernel_size=self.kernel, padding='valid', dilation_rate=[1,1]   ))
            #self.dense_z.append( tf.keras.layers.Dense(4,use_bias=False) )

    self.init_dense = tf.keras.layers.Dense(2*self.filters, activation=None) #tanh


  def waveLayer(self, x, counter):

    dil = self.dilations[counter % len(self.dilations) ]

    xin=tf.pad( x, [ [0,0], [(self.kernel[0]-1)//2,(self.kernel[0]-1)//2], [0, dil*(self.kernel[1]-1)],[0,0]] , "CONSTANT", 0.)
    y1 = self.conv_y1[counter](xin)
    y2 = self.conv_y2[counter](xin)
    
    #y1 = tf.keras.layers.Activation('tanh')(y1)
    #y2 = tf.keras.layers.Activation('sigmoid')(y2)
    y1 = dfh_tanh(y1)
    y2 = dfh_sigmoid(y2)

    z = y1*y2
    z=tf.pad( z, [ [0,0], [(self.kernel[0]-1)//2,(self.kernel[0]-1)//2], [0, (self.kernel[1]-1)],[0,0]] , "CONSTANT", 0.)
    z = self.conv_z0[counter](z)

    return z

  def call(self, x):

    x = self.init_dense(x)
    x = dfh_tanh(x)
    #
    skips=[]
    for b in range(self.blocks):
        for dd in range(len(self.dilations)):
            xw=self.waveLayer(x,dd + len(self.dilations)*b )
            skips.append(xw)
            x = xw + x

    x = tf.math.reduce_sum(tf.stack( skips, axis=-1), axis=-1)
    #x = tf.keras.layers.Activation("tanh")(x)
    x = dfh_tanh(x)
    
    return x  

class CombinedFIDNet(tf.keras.Model):
  def __init__(self, fidnet_filters_1h=8, blocks=3, fidnet_filters_13c=8, fidnet_kernel=(5,8), \
               refine_kernel=(9,9), refine_steps=4, rate=None ):
    super().__init__()

    self.fidnet_filters_1h = fidnet_filters_1h
    self.fidnet_filters_13c= fidnet_filters_13c
    self.refine_steps=refine_steps
    self.rate = rate
    
    self.fidnet_h = FIDNetLayer(filters=fidnet_filters_1h,  \
                                blocks=blocks,              \
                                kernel=fidnet_kernel        \
                                )

    self.fidnet_c = FIDNetLayer(filters=fidnet_filters_13c, \
                                blocks=blocks,              \
                                kernel=fidnet_kernel        \
                                )
    #
    # final conv2D layers
    self.conv_ref_r=[]
    self.conv_ref_t=[]
    for i in range(self.refine_steps):
        self.conv_ref_r.append( tf.keras.layers.Conv2D(filters=fidnet_filters_1h*4, kernel_size=refine_kernel, padding='same', dilation_rate=[1,1] ))
        self.conv_ref_t.append( tf.keras.layers.Conv2D(filters=fidnet_filters_13c*4, kernel_size=refine_kernel, padding='same', dilation_rate=[1,1] ))
    self.dense   = tf.keras.layers.Dense( 2, use_bias=True, activation=None)
    if rate is not None:
        self.dropout0 = tf.keras.layers.Dropout(rate)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, inputs, training=False):
    times, times_in, inp_2d_nc, inp_2d_c = inputs
    #
    # times:      (batch, coup, 512)
    # times_in:   (batch, 400)
    # inp_2d_nc:  (batch, coup, 13c:time, 1h:freq)
    # inp_2d_c:   (batch, coup, 13c:time, 1h:freq)    
    # 
    #
    # Make planes
    times_plane    = tf.tile( times[:,:1,:], (1,times_in.shape[-1],1)) #we only have one coupling
    times_plane    = tf.expand_dims( times_plane, axis=1) #(batch,coup,13c, 1h )
        
    times_in_plane = tf.expand_dims(tf.tile( tf.expand_dims(times_in, axis=-1), (1,1,times.shape[-1])), axis=1) #(batch,coup,13c, 1h)
    
    inp_2d_nc      = tf.transpose( inp_2d_nc, perm=(0,3,2,1)) # (batch, 1h:freq, 13c:time, coup)
    inp_2d_c       = tf.transpose( inp_2d_c , perm=(0,3,2,1)) #
    times_plane    = tf.transpose( times_plane, perm=(0,3,2,1)) #
    times_in_plane = tf.transpose( times_in_plane, perm=(0,3,2,1))    
    #
    inp = tf.concat([inp_2d_nc, inp_2d_c, times_plane, times_in_plane ], axis=-1)  #(batch, 1h, 13c, coups(h) )
    # 
    cout = self.fidnet_c(inp)
    if self.rate is not None:
        cout = self.dropout0(cout, training=training)  # (bathc, 1h, 400, 2*filters_13c)
    #
    # Now FT and transpose (cast to float32 and then back)
    cout = tf.cast( cout, tf.float32)
    cout = tf.transpose( cout, perm=(0,3,1,2) ) # (batch, filters, 1h, {13c.real, 13c.imag})
    #
    # Zero fill
    cout = tf.pad( cout, [ [0,0],[0,0],[0,0],[0,cout.shape[-1]]], "CONSTANT", constant_values=0.)
    multiplier = tf.concat( [ tf.constant([0.5,], dtype=cout.dtype), tf.ones( (cout.shape[-1]//2-1,), dtype=cout.dtype)], axis=0)
    multiplier = tf.complex( tf.reshape( multiplier, (1,1,-1)), tf.constant(0.,dtype=multiplier.dtype))
    #
    cout_ft= tf.signal.fftshift( tf.signal.fft( \
                                                multiplier*tf.complex(cout[:,:,:,0::2], cout[:,:,:,1::2]) \
    ) , axes=-1)
    #
    # HT 1H
    cout_ft = tf.transpose( tf.math.real(cout_ft), perm=(0,1,3,2)) #(batch, filters, 13C:freq, 1H:freq)
    cout_ft = hilbert(cout_ft)
    #
    # Inverse FT
    hinp = tf.math.conj(tf.signal.fft( tf.signal.fftshift( cout_ft, axes=-1))/tf.constant(cout_ft.shape[-1], dtype=cout_ft.dtype))
    multiplier = tf.pad( multiplier, [ [0,0], [0,0],[0, hinp.shape[-1] - multiplier.shape[-1]] ], "CONSTANT", constant_values=1.0)
    hinp = hinp / multiplier
    #
    # Zero points that were not used originally - take care with first point
    # time is (batch, coup, td)    
    zero_mask = tf.where( tf.abs(times[:,0,2::2]) > 0., 1., 0. ) # first two points are zero {0.r, 0.i, 1.r, ... }
    zero_mask = tf.pad( zero_mask, [ [0,0], [1,0] ], "CONSTANT", constant_values=1.0 )
    zero_mask = tf.complex( tf.expand_dims( zero_mask, axis=1), 0.)
    zero_mask = tf.expand_dims( zero_mask, axis=1)

    hinp = hinp[...,:hinp.shape[-1]//2] * zero_mask  # (Bathc, filters, 13c:freq(400), 1h:time(256) )
    #
    #                                                                             b,                 filters, 13c, 1h
    hinp = tf.reshape(tf.stack([tf.math.real(hinp),tf.math.imag(hinp)],axis=-1),(-1,2*self.fidnet_filters_13c,inp_2d_nc.shape[-2],inp_2d_nc.shape[-3]))

    hinp = tf.transpose( hinp, perm=(0,2,3,1))  # (batch, 13c, 1h, filters )
    
    #
    # re-cast to original dtype
    hinp = tf.cast( hinp, inp.dtype)
    hout = self.fidnet_h(hinp) #(batch, 13c(freq), 1h(time), filters)
    if self.rate is not None:
        hout = self.dropout1(hout, training=training)
        
    hout = tf.concat([hout,hinp], axis=-1)
    hout = tf.transpose( hout, perm=(0, 3, 1, 2)) # (batch, filters, 13c(freq), 1h(time))
    #
    # cast to float32 for FT
    hout = tf.cast( hout, tf.float32 )
    #
    #
    # Zero fill
    hout    = tf.pad( hout, [ [0,0],[0,0],[0,0],[0,hout.shape[-1]]], "CONSTANT", constant_values=0. )
    windowH = tf.concat( [ tf.constant([0.5], dtype=tf.dtypes.float32), tf.ones( hout.shape[-1]//2-1, dtype=tf.dtypes.float32)], axis=0)
    windowH = tf.expand_dims( windowH, axis=0)
    windowH = tf.complex( windowH, 0.)
    hout_ft= tf.signal.fftshift( tf.signal.fft( windowH * tf.complex(hout[...,0::2],hout[...,1::2]) ) ,axes=-1)
    hout_ft= tf.math.real( hout_ft )
    #
    hout_ft= tf.cast(hout_ft, inp.dtype)
    
    hout_ft= tf.transpose( hout_ft, perm=(0,2,3,1)) # (batch, 13C:freq, 1H:freq, filters)
    #
    # refinement
    #ref = self.dense(cout_ft)
    ref = hout_ft
    fnorm = tf.reduce_max( tf.abs(ref), axis=(1,2,3), keepdims=True )
    ref = ref / fnorm
    ref = tf.tile( ref, (1,1,1,2)) # This is doubling to get uncertainties and that ref layers have 4*filters
    #
    for i in range(self.refine_steps):
        rr = dfh_relu(self.conv_ref_r[i](ref))
        tt = dfh_tanh(self.conv_ref_t[i](ref))
        ref = tf.concat([tt,rr], axis=-1) + ref
    ref *= fnorm
    if self.rate is not None:
        ref = self.dropout2(ref, training=training)
    #
    # combine with spectra before refinement
    # This is now essentially [..., FIDNet(13C), FIDNet(1H), Refinement() ], which
    # should help the minimiser
    hout_final = tf.concat([ ref, hout_ft], axis=-1) #
    hout_final = self.dense(hout_final) # (...,{value, confidence~ } ). confidence will be sigmoid(confidence~)
    #
    hout_final = tf.stack( [hout_final[...,0], hout_final[...,1]/fnorm[...,0]], axis=-1)
    hout_final = tf.cast( hout_final, tf.float32)

    return hout_final

    
