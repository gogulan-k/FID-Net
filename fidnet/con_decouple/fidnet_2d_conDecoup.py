#!/usr/bin/python

# Gogulan Karunanithy, UCL, 2021
# Python code for decoupling 2D 'in-phase' CON spectra


import copy
import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import nmrglue as ng
import numpy as np
import tensorflow as tf
from tensorflow import keras


def read_data(infile):
    dic, data = ng.pipe.read(infile)
    # udic = ng.pipe.guess_udic(dic, data)

    return dic, data


def build_model_wavenet_large(blocks=3, num_filters=64):
    def ft_layer(x):
        ft = tf.transpose(x, perm=[0, 2, 1, 3])
        ft = tf.reshape(ft, [-1, 4, 512, 2])
        ft = tf.complex(ft[:, :, :, 0], ft[:, :, :, 1])
        ft = keras.layers.Lambda(tf.signal.fft)(ft)
        ft = tf.transpose(ft, perm=[0, 2, 1])
        ft = tf.math.real(ft)
        ft = keras.layers.Activation("linear", dtype="float32")(ft)
        return ft

    def waveLayer(x, num_filters, dil):
        y1 = keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=[8, 4],
            padding="same",
            dilation_rate=[dil, 1],
        )(x)

        y2 = keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=[8, 4],
            padding="same",
            dilation_rate=[dil, 1],
        )(x)

        y1 = keras.layers.Activation("tanh")(y1)
        y2 = keras.layers.Activation("sigmoid")(y2)

        z = y1 * y2
        z = keras.layers.Conv2D(
            filters=num_filters * 2, kernel_size=[8, 4], padding="same"
        )(z)

        return keras.layers.Add()([z, x]), z

    dilations = [1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64]
    input = keras.layers.Input(shape=[1024, 4, 1])
    x = input
    skips = []

    for dil in dilations * blocks:
        x, skip = waveLayer(x, num_filters, dil)
        skips.append(skip)

    x = keras.layers.Activation("relu")(keras.layers.Add()(skips))
    x = keras.layers.Conv2D(
        num_filters, kernel_size=[8, 4], padding="same", activation="relu"
    )(x)
    fin = keras.layers.Conv2D(
        1, kernel_size=[8, 4], padding="same", activation="tanh", dtype="float32"
    )(x)
    ft_fin = ft_layer(fin)
    model = keras.Model(inputs=[input], outputs=[fin, ft_fin])
    model.compile(
        loss=["mse", "mse"],
        loss_weights=[0.0, 1.0],
        optimizer=keras.optimizers.RMSprop(learning_rate=1.0e-4),
    )

    return model


def direct_decouple(model_weights, infile, outfile, shift=True, f1180=False):
    model = build_model_wavenet_large()
    model.load_weights(model_weights)

    dic, data = read_data(infile)

    # udic = ng.pipe.guess_udic(dic, data)

    if data.ndim != 2:
        print("input data must be 2D")
        print("aborting now...")
        sys.exit()

    dic_dl = copy.deepcopy(dic)

    if data.shape[1] > 512:
        print("direct dimension has more than 512 complex points")
        print("FID will be truncated at 512 complex points to run DNN")
        data = data[:, :512]

    real_new = np.zeros((data.shape[0], 2 * data.shape[1]))
    real_new[:, ::2] = np.real(data)
    real_new[:, 1::2] = np.imag(data)
    data = copy.deepcopy(real_new)

    Hpoints = data.shape[0]
    Cpoints = data.shape[1]
    dic_dl["FDSIZE"] = Cpoints // 2
    dic_dl["FDF2APOD"] = Cpoints // 2
    dic_dl["FDF2TDSIZE"] = Cpoints // 2

    full_max = np.max(data)
    data = data / full_max

    samp_av, scale = setup_2d_plane(data, 1024)

    res = model.predict(samp_av)

    res = tf.convert_to_tensor(res[0])
    res = rescale_dat(res, scale, 1024)
    res = get_average_results(res, data.shape[0], 1024)
    res = res[:, :Cpoints, :, 0]
    data_write = res[0, :, :].numpy()
    data_write = data_write[0::2, :] + 1.0j * data_write[1::2, :]
    data_write = np.transpose(data_write)
    data_write *= full_max

    ng.pipe.write(outfile, dic_dl, data_write, overwrite=True)

    data = tf.convert_to_tensor(data)
    data = tf.expand_dims(data, axis=0)
    data = tf.transpose(data, perm=[0, 2, 1])

    data_ft = ft_second(
        data, npoints1=Hpoints, npoints2=Cpoints, f1180=False, shift=shift
    )
    data_ft = data_ft / tf.reduce_max(data_ft)

    # dat_plot = data / tf.reduce_max(data)
    # res_plot = res / tf.reduce_max(res)

    res_ft = ft_second(
        res, npoints1=Hpoints, npoints2=Cpoints, f1180=False, shift=shift
    )
    res_ft = res_ft / tf.reduce_max(res_ft)

    plot_lvl1 = 0.10
    plot_lvl2 = 0.06
    levy1 = getLevels(np.max(data_ft) * plot_lvl1, 1.3, 14)
    levy2 = getLevels(np.max(res_ft) * plot_lvl2, 1.3, 14)

    ax1 = plt.subplot(2, 1, 1)
    plot_contour(
        ax1,
        data_ft,
        lvl=levy1,
        col1="red",
        col2="teal",
        invert=False,
        invert_x=True,
        transpose=True,
    )

    ax2 = plt.subplot(2, 1, 2)
    plot_contour(
        ax2,
        res_ft,
        lvl=levy2,
        col1="green",
        col2="blueviolet",
        invert=False,
        invert_x=True,
        transpose=True,
    )

    plt.savefig(outfile + ".png")

    plt.close('all')


def ft_second(ft, npoints1=128, npoints2=100, f1180=False, shift=False, smile=False):
    if not smile:
        ft = tf.transpose(ft, perm=[0, 2, 1])
        ft = tf.reshape(ft, [1, npoints1, npoints2 // 2, 2])
        ft = tf.complex(ft[..., 0], ft[..., 1])

    ft = np.array(ft)

    ft = ng.proc_base.sp(ft, off=0.45, end=0.98, pow=2.0, inv=False, rev=False)
    ft = ng.proc_base.zf(ft, npoints2 // 2)
    if not f1180:
        ft[..., 0] = ft[..., 0] * 0.5
    ft = tf.convert_to_tensor(ft)
    if shift:
        ft = tf.signal.fftshift(tf.signal.fft(ft), axes=2)
    else:
        ft = tf.signal.fft(ft)
    if f1180:
        ft = np.array(ft)
        ft = ng.proc_base.ps(ft, p0=90.0, p1=-180.0, inv=False)
        ft = tf.convert_to_tensor(ft)

    ft = tf.transpose(ft, perm=[0, 2, 1])
    ft = tf.math.real(ft)

    return ft


def getLevels(min, fac, num):
    return np.array([min * (fac**i) for i in range(num)])


def setup_2d_plane(ft1_samp, tot):
    # input 2d plane
    # output something that is happy to go into model.predict
    ft1_samp = tf.convert_to_tensor(ft1_samp)
    padding_recon = [[3, 3], [0, tot - tf.shape(ft1_samp)[1]]]
    samp_av = tf.pad(ft1_samp, padding_recon, "Constant", constant_values=0.0)
    scale = np.array(
        [
            np.max(np.fabs(samp_av[i : i + 4, :]))
            for i in range(tf.shape(ft1_samp)[0] + 3)
        ]
    )

    sampy = np.zeros((scale.shape[0], 4, tf.shape(samp_av)[1]))
    for i in range(scale.shape[0]):
        sampy[i, :, :] = samp_av[i : i + 4, :]
    samp_av = tf.convert_to_tensor(sampy)
    samp_av = tf.transpose(samp_av, perm=[0, 2, 1])
    samp_av = tf.transpose(samp_av, perm=[2, 1, 0])

    samp_av = samp_av / scale
    samp_av = tf.transpose(samp_av, perm=[2, 1, 0])
    samp_av = tf.expand_dims(samp_av, axis=3)

    return samp_av, scale


def get_average_results(dat, Hpoints, tot):
    print("in shape...,", tf.shape(dat))
    res_div = np.zeros((tot, Hpoints), dtype=np.float32)
    for i in range(Hpoints):
        ind = 4 * i + 3
        res_div[:, i] = 0.25 * (
            dat[0, :, ind, 0]
            + dat[0, :, ind + 3, 0]
            + dat[0, :, ind + 6, 0]
            + dat[0, :, ind + 9, 0]
        )

    res_div = tf.convert_to_tensor(res_div)
    res_div = tf.expand_dims(res_div, axis=0)
    res_div = tf.expand_dims(res_div, axis=3)

    return res_div


def rescale_dat(dat, scale, tot):
    dat = tf.transpose(dat, perm=[3, 1, 2, 0])
    dat = dat * scale
    dat = tf.transpose(dat, perm=[3, 2, 1, 0])
    dat = tf.reshape(dat, [1, -1, tot, 1])
    dat = tf.transpose(dat, perm=[0, 2, 1, 3])
    return dat


def plot_contour(
    ax,
    ft_outer,
    col1="blues",
    col2="reds",
    lvl=None,
    invert=False,
    invert_x=False,
    transpose=True,
):
    ft_outer = ft_outer.numpy()[0, :, :]
    if transpose:
        ft_outer = np.transpose(ft_outer)

    if lvl is None:
        lvl = getLevels(np.max(ft_outer) * 0.05, 1.22, 20)

    ax.contour(ft_outer, levels=lvl, colors=col1)
    ax.contour(-1.0 * ft_outer, levels=lvl, colors=col2)

    if invert:
        ax.invert_yaxis()
    if invert_x:
        ax.invert_xaxis()
