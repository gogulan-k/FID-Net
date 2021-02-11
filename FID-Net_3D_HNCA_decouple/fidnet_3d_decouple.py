#!/usr/bin/python
# Gogulan Karunanithy, UCL, 2021
# Code for performing 3D decoupling of HNCA and HN(CO)CA spectra using FID-Net

MODEL_WEIGHTS = '../FID-Net_modelWeights/fidnet_3dca_decouple.h5'
# this should be changed to the absolute path of the downloaded weights file

import tensorflow as tf
import copy, sys
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
    padding_recon = [[3,3],[0,512 - tf.shape(ft1_samp)[1]]]
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

def remake_dic(in_dic,Cpoints):
    new_dic = copy.deepcopy(in_dic)
    new_dic['FDF3TDSIZE'] = Cpoints//2
    new_dic['FDF3SIZE'] = Cpoints
    new_dic['FDF3apod'] = Cpoints//2
    return new_dic

def decouple_spec(model_weights_jcoup,file,outfile):
    model_jcoup = build_model(num_blocks = 3, num_filters = 32)
    model_jcoup.load_weights(model_weights_jcoup)

    dic,data = ng.pipe.read(file)
    Cpoints = data.shape[0]

    if Cpoints>512:
        print('This FID-Net can deal with a maximum of 256 complex points')
        print('truncating the 13C dimension at 256 complex points')
        data = data[:512,:,:]
        Cpoints = 512

    data_fin = np.zeros_like(data)

    data = np.transpose(data, axes = [1,2,0])

    dic_dl = remake_dic(dic, Cpoints)
    full_max = np.max(data)


    for k in range(data.shape[0]):
        print('doing plane ', k+1, ' of ', data.shape[0])
        ft1_samp = data[k,:,:]

        samp_av, scale = setup_2d_plane(ft1_samp)

        res = model_jcoup.predict(samp_av)

        res = tf.convert_to_tensor(res[0])
        res = rescale_dat(res,scale)
        res = get_average_results(res, ft1_samp.shape[0])
        res = res[:,:Cpoints,:,0]
        data_fin[:,k,:] = res.numpy()[0,:,:]

    data_fin *= full_max
    ng.pipe.write(outfile,dic_dl, data_fin,overwrite=True)

import argparse
parser = argparse.ArgumentParser(description='FID-Net 3D HNCA decoupling')
parser.add_argument('-in','--in', help='Input  spectra. This is a 3D HNCA or \
                    HN(CO)CA spectra with the 13C dimension in the time domain. \
                    The 15N and 1H dimensions should be phased and Fourier transformed. \
                    The order of the input dimensions must be 1H,15N, 13C.', required=True)
parser.add_argument('-out','--out', help='Name of output spectra. Defaults to \
                        decouple.ft2', required=False, default = 'decouple.ft2')
args = vars(parser.parse_args())

decouple_spec(MODEL_WEIGHTS,args['in'],args['out'])
