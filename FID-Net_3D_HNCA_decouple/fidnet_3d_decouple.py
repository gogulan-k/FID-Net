import tensorflow as tf
import os, copy, sys
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import nmrglue as ng

def build_model(num_blocks = 3, num_filters = 64):
    def ft_layer(x):
        ft = tf.transpose(x, perm = [0,2,1,3])
        ft = tf.reshape(ft, [-1,4,256,2])
        ft = tf.complex(ft[:,:,:,0],ft[:,:,:,1])
        ft = keras.layers.Lambda(tf.signal.fft)(ft)
        ft = tf.transpose(ft, perm = [0,2,1])
        ft = tf.math.real(ft)
        return ft

    def waveLayer(x,num_filters,dil):
        y1 = keras.layers.Conv2D(filters = num_filters, kernel_size=[8,4],
                                padding="same", dilation_rate=[dil,1])(x)

        y2 = keras.layers.Conv2D(filters = num_filters, kernel_size=[8,4],
                                    padding="same", dilation_rate=[dil,1])(x)

        y1 = keras.layers.Activation('tanh')(y1)
        y2 = keras.layers.Activation('sigmoid')(y2)

        z = y1*y2
        z =  keras.layers.Conv2D(filters = num_filters*2, kernel_size=[8,4], padding="same")(z)

        return keras.layers.Add()([z,x]), z

    dilations = [1,2,4,6,8,10,12,14,16,20,24,28,32]


    input =  keras.layers.Input(shape=[512, 4, 1])
    x = input
    skips = []

    for dil in dilations*num_blocks:
        x, skip = waveLayer(x, num_filters, dil)
        skips.append(skip)

    x = keras.layers.Activation("relu")(keras.layers.Add()(skips))
    x = keras.layers.Conv2D(num_filters, kernel_size=[8,4],padding="same", activation="relu")(x)
    fin = keras.layers.Conv2D(1, kernel_size=[8,4], padding="same", activation="tanh")(x)
    ft_fin = ft_layer(fin)
    model = keras.Model(inputs=[input], outputs=[fin,ft_fin])
    model.compile(loss=["mse","mse"], loss_weights = [0.0,1.0],
                    optimizer=keras.optimizers.RMSprop(lr=1.0e-4))

    return model


def setup_2d_plane(ft1_samp):
    # input 2d plane
    # output something that is happy to go into model.predict
    ft1_samp = tf.convert_to_tensor(ft1_samp)
    padding_recon = [[3,3],[0,0]]
    samp_av = tf.pad(ft1_samp, padding_recon, 'Constant', constant_values = 0.0)
    scale = np.array([np.max(np.fabs(samp_av[i:i+4,:])) for i in range((tf.shape(ft1_samp)[0]+3))])
    sampy = np.zeros((scale.shape[0], 4, tf.shape(samp_av)[1]))
    for i in range(scale.shape[0]):
        sampy[i,:,:] = samp_av[i:i+4,:]
    samp_av = tf.convert_to_tensor(sampy)
    samp_av = tf.transpose(samp_av, perm = [0,2,1])
    samp_av = tf.transpose(samp_av, perm = [2,1,0])

    samp_av = samp_av/scale
    samp_av = tf.transpose(samp_av, perm = [2,1,0])
    samp_av = tf.expand_dims(samp_av,axis=3)

    return samp_av, scale

def get_average_results(dat,Hpoints):
    print('in shape...,', tf.shape(dat))
    res_div = np.zeros((512,Hpoints), dtype = np.float32)
    for i in range(Hpoints):
        ind = 4*i + 3
        res_div[:,i] = 0.25*(dat[0,:,ind,0]+dat[0,:,ind+3,0]+dat[0,:,ind+6,0]+dat[0,:,ind+9,0])

    res_div = tf.convert_to_tensor(res_div)
    res_div = tf.expand_dims(res_div,axis=0)
    res_div = tf.expand_dims(res_div,axis=3)

    return res_div


def rescale_dat(dat,scale):
    dat = tf.transpose(dat, perm=[3,1,2,0])
    dat = dat*scale
    dat = tf.transpose(dat, perm=[3,1,2,0])
    dat = tf.transpose(dat, perm=[0,2,1,3])
    dat = tf.reshape(dat, [1,-1,512,1])
    dat = tf.transpose(dat, perm=[0,2,1,3])
    return dat


def decouple_spec(file, model_weights_jcoup):
    model_jcoup = build_model(num_blocks = 3, num_filters = 32)
    model_jcoup.load_weights(model_weights_jcoup)

    dic,data = ng.pipe.read(file)
    data_fin = np.zeros_like(data)
    print(data.shape)
    print(data_fin.shape)
    data = np.transpose(data, axes = [1,2,0])
    np1_spec = data.shape[1]


    data_samp = copy.deepcopy(data)

    full_max = np.max(data)
    full_max_samp = np.max(data_samp)

    data = data/full_max
    data_samp = data_samp/full_max_samp

    for k in range(data.shape[0]):
        print('doing plane ', k+1, ' of ', data.shape[0])
        ft1_samp = data_samp[k,:,:]


        plotty_ft1 = tf.expand_dims(data[k,:,:], axis = 0)
        samp_av, scale = setup_2d_plane(ft1_samp)

        res = model_jcoup.predict(samp_av)

        res = tf.convert_to_tensor(res[0])
        res = rescale_dat(res,scale)
        res = get_average_results(res, ft1_samp.shape[0])
        res = res[:,:512,:,0]
        data_fin[:,k,:] = res.numpy()[0,:,:]


    ng.pipe.write('test_decouple.ft2',dic, data_fin,overwrite=True)


decouple_spec('example/T4L.ft2', '../FID-Net_modelWeights/fidnet_3dca_decouple.h5')
