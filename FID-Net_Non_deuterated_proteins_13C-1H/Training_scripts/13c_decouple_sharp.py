#!/usr/bin/python

# Gogulan Karunanithy, UCL, 2023

import tensorflow as tf
import time
import os, copy
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.mixed_precision import experimental as mixed_precision


policy = mixed_precision.Policy('mixed_float16')
#policy = mixed_precision.Policy('float32')
mixed_precision.set_policy(policy)

def parse_function(example_proto):
    # initially these all exist as tf strings that need to be converted
    # to bytes: ie here we deserealise the tensor objects
    feat_des = {
    "nsigs": tf.io.FixedLenFeature([], tf.string),
    "p0": tf.io.FixedLenFeature([], tf.string),
    "p1": tf.io.FixedLenFeature([], tf.string),
    "amp_cent": tf.io.FixedLenFeature([], tf.string),
    "amp_dist": tf.io.FixedLenFeature([], tf.string),
    "sig_mask": tf.io.FixedLenFeature([], tf.string),

    "npoints_1": tf.io.FixedLenFeature([], tf.string),
    "sw_1": tf.io.FixedLenFeature([], tf.string),
    "freq_1": tf.io.FixedLenFeature([], tf.string),
    "r2_1": tf.io.FixedLenFeature([], tf.string),

    "npoints_2": tf.io.FixedLenFeature([], tf.string),
    "sw_2": tf.io.FixedLenFeature([], tf.string),
    "freq_2": tf.io.FixedLenFeature([], tf.string),
    "r2_2": tf.io.FixedLenFeature([], tf.string),

    "Jcoup": tf.io.FixedLenFeature([], tf.string)
    }
    parsed_example = tf.io.parse_single_example(example_proto,feat_des)

    for key in parsed_example:
        # here we deserealise the tensor values
        if key == 'npoints_1' or key == 'npoints_2' or key == 'nsigs':
            parsed_example[key]=tf.io.parse_tensor(parsed_example[key],
                                                    tf.dtypes.int64)
        elif key == 'sig_mask':
            parsed_example[key] = tf.io.parse_tensor(parsed_example[key],
                                                    tf.dtypes.bool)
        else:
            parsed_example[key] = tf.io.parse_tensor(parsed_example[key],
                                                        tf.dtypes.float32)

    return parsed_example

def read_tfrecord(infile):
    raw_dataset =  tf.data.TFRecordDataset([infile])
    parsed_dataset = raw_dataset.map(parse_function)
    proc_dataset = parsed_dataset.map(process_example)

    return proc_dataset

def process_example(pdic):

    nsigs = pdic['nsigs']
    p0 = pdic['p0']
    p1 = pdic['p1']
    sig_mask = pdic['sig_mask']
    sig_mask.set_shape([None])
    pdic['amp_dist'] = tf.boolean_mask(pdic['amp_dist'],sig_mask)
    amp_cent= pdic['amp_cent']
    amp_dist = pdic['amp_dist']

    npoints_1 = pdic['npoints_1']
    sw_1 = pdic['sw_1']
    dw_1 = tf.divide(tf.constant(1.0), sw_1)
    pdic['freq_1'] = tf.boolean_mask(pdic['freq_1'],sig_mask)
    pdic['r2_1'] = tf.boolean_mask(pdic['r2_1'],sig_mask)
    freq_1 = pdic['freq_1']
    r2_1 = pdic['r2_1']

    npoints_2 = pdic['npoints_2']
    sw_2 = pdic['sw_2']
    dw_2 = tf.divide(tf.constant(1.0), sw_2)
    pdic['freq_2'] = tf.boolean_mask(pdic['freq_2'],sig_mask)
    pdic['r2_2'] = tf.boolean_mask(pdic['r2_2'],sig_mask)
    freq_2 = pdic['freq_2']
    r2_2 = pdic['r2_2']

    r2_2_max = 25.0
    r2_2_lim = tf.maximum(r2_2_max*tf.math.tanh(r2_2/r2_2_max),r2_2_max*(1-tf.math.tanh(r2_2/r2_2_max)))
    pdic['Jcoup'] = tf.boolean_mask(pdic['Jcoup'],sig_mask)
    Jcoup = pdic['Jcoup']

    probs = tf.random.uniform(shape =[nsigs],minval = 0.0, maxval = 1.0)
    Jcoup = tf.where(probs<0.95, Jcoup, 0.0)

    pi2 = tf.constant(2.0*3.14159265)

    times_1 = tf.range(tf.cast(npoints_1,dtype=tf.float32))*dw_1
    times_2 = tf.range(tf.cast(npoints_2,dtype=tf.float32))*dw_2

    # calculate fid
    fid1 =  tf.transpose(tf.multiply(
             tf.transpose(tf.exp(
                        tf.tensordot(
                            tf.complex(
                                tf.multiply(tf.constant(-1.0),r2_1),
                                tf.multiply(pi2,freq_1)),
                            tf.complex(times_1, tf.constant(0.0)), axes= 0 ))),
                         tf.cast(amp_dist, tf.complex64)))


    fid2_1 =  tf.transpose(tf.multiply(
                    tf.transpose(tf.exp(
                        tf.tensordot(
                            tf.complex(
                                tf.multiply(tf.constant(-1.0),r2_2),
                                tf.multiply(pi2,freq_2-0.5*Jcoup)),
                                tf.complex(times_2, tf.constant(0.0)), axes= 0 ))),
                            tf.cast(amp_dist, tf.complex64)))

    fid2_2 =  tf.transpose(tf.multiply(
                    tf.transpose(tf.exp(
                        tf.tensordot(
                            tf.complex(
                                tf.multiply(tf.constant(-1.0),r2_2),
                                tf.multiply(pi2,freq_2+0.5*Jcoup)),
                                tf.complex(times_2, tf.constant(0.0)), axes= 0 ))),
                            tf.cast(amp_dist, tf.complex64)))

    fid2_jcoup = 0.5*(fid2_1+fid2_2)

    fid2_norm =     tf.transpose(tf.multiply(
                tf.transpose(tf.exp(
                    tf.tensordot(
                        tf.complex(
                            tf.multiply(tf.constant(-1.0),r2_2_lim),
                            tf.multiply(pi2,freq_2)),
                            tf.complex(times_2, tf.constant(0.0)), axes= 0 ))),
                        tf.cast(amp_dist, tf.complex64)))

    fid_r_jcoup = tf.reduce_sum(tf.einsum('...j,...k->...jk',
                        tf.complex(tf.math.real(fid2_jcoup),0.0),  fid1), axis=0)
    fid_i_jcoup = tf.reduce_sum(tf.einsum('...j,...k->...jk',
                        tf.complex(tf.math.imag(fid2_jcoup),0.0) , fid1), axis=0)


    fid_r_norm = tf.reduce_sum(tf.einsum('...j,...k->...jk',
                        tf.complex(tf.math.real(fid2_norm),0.0),  fid1), axis=0)
    fid_i_norm = tf.reduce_sum(tf.einsum('...j,...k->...jk',
                        tf.complex(tf.math.imag(fid2_norm),0.0) , fid1), axis=0)

    # fourier transform
    ft_r_jcoup = tf.signal.fftshift(tf.signal.fft(fid_r_jcoup),axes=-1)
    ft_i_jcoup = tf.signal.fftshift(tf.signal.fft(fid_i_jcoup),axes=-1)

    ft_r_norm = tf.signal.fftshift(tf.signal.fft(fid_r_norm),axes=-1)
    ft_i_norm = tf.signal.fftshift(tf.signal.fft(fid_i_norm),axes=-1)

    # delete imaginary parts
    ft_r_jcoup = tf.math.real(ft_r_jcoup)
    ft_i_jcoup = tf.math.real(ft_i_jcoup)

    ft_r_norm = tf.math.real(ft_r_norm)
    ft_i_norm = tf.math.real(ft_i_norm)

    ft_r_orig_jcoup = ft_r_jcoup
    ft_i_orig_jcoup = ft_i_jcoup

    ft_r_orig_norm = ft_r_norm
    ft_i_orig_norm = ft_i_norm

    # get 4 slices from this
    slice_rand = tf.random.uniform(
                        [1], minval=0,maxval=npoints_1-4,
                        dtype = tf.dtypes.int64)[0]



    ft_r_orig_jcoup = ft_r_jcoup[...,slice_rand:slice_rand+4]
    ft_i_orig_jcoup = ft_i_jcoup[...,slice_rand:slice_rand+4]

    ft_r_orig_norm = ft_r_norm[...,slice_rand:slice_rand+4]
    ft_i_orig_norm = ft_i_norm[...,slice_rand:slice_rand+4]

    # normalise
    max_val_jcoup =   tf.maximum(
                      tf.maximum(
                      tf.reduce_max(tf.abs(ft_r_orig_jcoup)), tf.reduce_max(tf.abs(ft_i_orig_jcoup))), 1e-6)

    max_val_norm =   tf.maximum(
                     tf.maximum(
                     tf.reduce_max(tf.abs(ft_r_orig_norm)), tf.reduce_max(tf.abs(ft_i_orig_norm))), 1e-6)


    ft_r_orig_jcoup = ft_r_orig_jcoup/max_val_jcoup
    ft_i_orig_jcoup = ft_i_orig_jcoup/max_val_jcoup

    ft_r_orig_norm = ft_r_orig_norm/max_val_norm
    ft_i_orig_norm = ft_i_orig_norm/max_val_norm
    # zero fill setup
    padding_2 = [[0,512-tf.shape(ft_r_jcoup)[0]],[0,0]]

    ft_r_orig_norm = tf.pad(ft_r_orig_norm, padding_2, 'Constant', constant_values = 0.0)
    ft_i_orig_norm = tf.pad(ft_i_orig_norm, padding_2, 'Constant', constant_values = 0.0)

    # this gives us no noise


    ft1_orig_norm = tf.reshape(tf.stack([ft_r_orig_norm,ft_i_orig_norm],axis=1),
                                                [-1, tf.shape(ft_r_orig_norm)[1]])
    # add noise:
    noise_lvl = tf.random.uniform([1], minval = 0.001, maxval = 0.04,
                                            dtype = tf.dtypes.float32)
    noise1 = tf.random.normal([tf.shape(ft_r_orig_jcoup)[0],tf.shape(ft_r_orig_jcoup)[1]],
                             mean=0.0,stddev=noise_lvl, dtype=tf.dtypes.float32)
    noise2 = tf.random.normal([tf.shape(ft_r_orig_jcoup)[0],tf.shape(ft_r_orig_jcoup)[1]],
                             mean=0.0,stddev=noise_lvl, dtype=tf.dtypes.float32)

    ft_r_noise_jcoup = ft_r_orig_jcoup + noise1
    ft_i_noise_jcoup = ft_i_orig_jcoup + noise2

    remax = tf.maximum(
            tf.reduce_max(tf.abs(ft_r_noise_jcoup)), tf.reduce_max(tf.abs(ft_i_noise_jcoup)))
    ft_r_noise_jcoup = ft_r_noise_jcoup/remax
    ft_i_noise_jcoup = ft_i_noise_jcoup/remax



    ft_r_noise_jcoup_samp = tf.pad(ft_r_noise_jcoup, padding_2, 'Constant', constant_values = 0.0)
    ft_i_noise_jcoup_samp = tf.pad(ft_i_noise_jcoup, padding_2, 'Constant', constant_values = 0.0)


    ft1_jcoup_samp = tf.reshape(tf.stack([ft_r_noise_jcoup_samp,ft_i_noise_jcoup_samp],axis=1),
                                                [-1, tf.shape(ft_r_noise_jcoup_samp)[1]])

    ft_orig_norm = tf.transpose(ft1_orig_norm)
    ft_orig_norm = tf.reshape(ft_orig_norm, [4,512,2])
    ft_orig_norm = tf.complex(ft_orig_norm[...,0],ft_orig_norm[...,1])

    ft_orig_norm = tf.signal.fft(ft_orig_norm)
    ft_orig_norm = tf.transpose(ft_orig_norm)
    ft_orig_norm = tf.math.real(ft_orig_norm)



    ft1_jcoup_samp = tf.expand_dims(ft1_jcoup_samp, 2)
    ft1_orig_norm = tf.expand_dims(ft1_orig_norm, 2)
    ft_orig_norm = tf.expand_dims(ft_orig_norm, 2)

    return ft1_jcoup_samp, (ft1_orig_norm, ft_orig_norm)


def build_model_wavenet():

    def ft_layer(x):
        ft = tf.transpose(x, perm = [0,2,1,3])
        ft = tf.reshape(ft, [-1,4,512,2])
        ft = tf.complex(ft[:,:,:,0],ft[:,:,:,1])

        ft = keras.layers.Lambda(tf.signal.fft)(ft)
        ft = tf.transpose(ft, perm = [0,2,1])
        ft = tf.math.real(ft)
        ft = keras.layers.Activation('linear', dtype='float32')(ft)

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

    num_filters = 64
    blocks = 3

    dilations = [1,2,4,6,8,10,12,14,16,20,24,28,32,40,48,56,64]
    input =  keras.layers.Input(shape=[1024, 4, 1])
    x = input
    skips = []

    for dil in dilations*blocks:
        x, skip = waveLayer(x, num_filters, dil)
        skips.append(skip)

    x = keras.layers.Activation("relu")(keras.layers.Add()(skips))
    x = keras.layers.Conv2D(num_filters, kernel_size=[8,4],padding="same", activation="relu")(x)
    fin = keras.layers.Conv2D(1, kernel_size=[8,4], padding="same", activation="tanh", dtype = 'float32')(x)
    ft_fin = ft_layer(fin)
    model = keras.Model(inputs=[input], outputs=[fin,ft_fin])
    model.compile(loss=["mse","mse"], loss_weights = [0.0,1.0],
                    optimizer=keras.optimizers.RMSprop(lr=1.0e-4))

    return model



def get_run_logdir(root_logdir):

    run_id = time.strftime("/run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


def run_model(dtrain_file,dval_file):

    traind =  tf.data.TFRecordDataset([dtrain_file])
    traind = traind.shuffle(buffer_size=1000).repeat()
    traind = traind.map(parse_function,
                num_parallel_calls = tf.data.experimental.AUTOTUNE)
    traind = traind.map(process_example,
                num_parallel_calls = tf.data.experimental.AUTOTUNE)
    traind = traind.batch(BATCHSIZE, drop_remainder=True)
    traind = traind.prefetch(tf.data.experimental.AUTOTUNE)


    vald =  tf.data.TFRecordDataset([dval_file])
    vald = vald.shuffle(buffer_size=1000).repeat()
    vald = vald.map(parse_function,
                num_parallel_calls = tf.data.experimental.AUTOTUNE)
    vald = vald.map(process_example,
                num_parallel_calls = tf.data.experimental.AUTOTUNE)
    vald = vald.batch(BATCHSIZE, drop_remainder=True)
    vald = vald.prefetch(tf.data.experimental.AUTOTUNE)

    checkpoint_cb = keras.callbacks.ModelCheckpoint('fidnet_13c_methyl.h5',
                                        save_best_only = True)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                    restore_best_weights = True)
    root_logdir = os.path.join(os.curdir)
    run_logdir = get_run_logdir(root_logdir)
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

    with strategy.scope():
        model = build_model_wavenet()
        print(model.summary())

        print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
        K.set_value(model.optimizer.lr, 1.0e-5)
        print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))
        history = model.fit(traind,epochs=1000,
                    callbacks = [checkpoint_cb,early_stopping_cb],
                    validation_data = vald,
                    steps_per_epoch=TRAINSIZE//BATCHSIZE,
                    validation_steps = VALSIZE//BATCHSIZE)

strategy = tf.distribute.MirroredStrategy()
BATCHSIZE = 64
TRAINSIZE = 500000
VALSIZE = 50000

TRAINING_SET = 'training_data/methyl_13c_500k.tfrecord'
VALIDATION_SET = 'training_data/methyl_13c_50k.tfrecord'

run_model(TRAINING_SET,VALIDATION_SET)
