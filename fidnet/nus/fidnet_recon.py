#!/usr/bin/python
# Gogulan Karunanithy, UCL, 2021
# Code for performing reconstructions using FID-Net


import copy
import sys
import os 

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import nmrglue as ng
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

def build_model(num_blocks=3, num_filters=64):
    def ft_layer(x):
        ft = tf.transpose(x, perm=[0, 2, 1, 3])
        ft = tf.reshape(ft, [-1, 4, 256, 2])
        ft = tf.complex(ft[:, :, :, 0], ft[:, :, :, 1])
        ft = keras.layers.Lambda(tf.signal.fft)(ft)
        ft = tf.transpose(ft, perm=[0, 2, 1])
        ft = tf.math.real(ft)
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

    dilations = [1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32]

    input = keras.layers.Input(shape=[512, 4, 1])
    x = input
    skips = []

    for dil in dilations * num_blocks:
        x, skip = waveLayer(x, num_filters, dil)
        skips.append(skip)

    x = keras.layers.Activation("relu")(keras.layers.Add()(skips))
    x = keras.layers.Conv2D(
        num_filters, kernel_size=[8, 4], padding="same", activation="relu"
    )(x)
    fin = keras.layers.Conv2D(1, kernel_size=[8, 4], padding="same", activation="tanh")(
        x
    )
    ft_fin = ft_layer(fin)
    model = keras.Model(inputs=[input], outputs=[fin, ft_fin])
    model.compile(
        loss=["mse", "mse"],
        loss_weights=[0.0, 1.0],
        optimizer=keras.optimizers.RMSprop(learning_rate=1.0e-4),
    )

    return model


def rescale_dat(dat, scale):
    dat = tf.transpose(dat, perm=[3, 1, 2, 0])
    dat = dat * scale
    dat = tf.transpose(dat, perm=[3, 1, 2, 0])
    dat = tf.transpose(dat, perm=[0, 2, 1, 3])
    dat = tf.reshape(dat, [1, -1, 512, 1])
    dat = tf.transpose(dat, perm=[0, 2, 1, 3])
    return dat


def get_average_results(dat, Hpoints):
    res_div = np.zeros((512, Hpoints), dtype=np.float32)
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


def get_ind_spectra(
    dat, data_ft, Hpoints, Npoints, dl_dic, outfile, f1180=False, shift=False, flip=False
):
    spec1 = np.zeros((512, Hpoints), dtype=np.float32)
    spec2 = np.zeros((512, Hpoints), dtype=np.float32)
    spec3 = np.zeros((512, Hpoints), dtype=np.float32)
    spec4 = np.zeros((512, Hpoints), dtype=np.float32)
    for i in range(Hpoints):
        ind = 4 * i + 3
        spec1[:, i] = dat[0, :, ind, 0]
        spec2[:, i] = dat[0, :, ind + 3, 0]
        spec3[:, i] = dat[0, :, ind + 6, 0]
        spec4[:, i] = dat[0, :, ind + 9, 0]

    spec_all = [spec1, spec2, spec3, spec4]
    # rmsd = []

    for i, spec in enumerate(spec_all):
        spec = tf.convert_to_tensor(spec)
        spec = tf.expand_dims(spec, axis=0)
        spec = tf.expand_dims(spec, axis=3)
        spec = spec[:, :Npoints, :, 0]
        spec = ft_second(
            spec, npoints1=Hpoints, npoints2=Npoints, f1180=f1180, shift=shift
        )
        spec = spec / tf.reduce_max(spec)
        spec_all[i] = spec

    std_spec = np.std(
        (
            spec_all[0][0, :, :].numpy(),
            spec_all[1][0, :, :].numpy(),
            spec_all[2][0, :, :].numpy(),
            spec_all[3][0, :, :].numpy(),
        ),
        axis=0,
    )

    std_dic = copy.deepcopy(dl_dic)
    std_dic["FDF1TDSIZE"] = float(std_spec.shape[0])
    std_dic["FDF1APOD"] = float(std_spec.shape[0])
    std_dic["FDSLICECOUNT"] = float(std_spec.shape[0])
    std_dic["FDSPECNUM"] = float(std_spec.shape[0])
    std_dic["FDF1FTFLAG"] = 1.0
    std_dic["FDF1QUADFLAG"] = 1.0
    std_dic["FDQUADFLAG"] = 1.0

    if flip:
        std_spec = np.flip(std_spec, axis=0)

    ng.pipe.write(str(Path(outfile).parents[0] / "nus_std.ft2"), std_dic, std_spec, overwrite=True)


def plot_contour(ax, ft_outer, col="viridis", lvl=None, invert=False):
    ft_outer = ft_outer.numpy()[0, :, :]

    if lvl is None:
        lvl = getLevels(np.max(ft_outer) * 0.045, 1.2, 22)

    ax.contour(ft_outer, levels=lvl, cmap=col)

    if invert:
        ax.invert_yaxis()


def ft_second(ft, npoints1=128, npoints2=100, f1180=True, shift=True):
    ft = tf.transpose(ft, perm=[0, 2, 1])
    ft = tf.reshape(ft, [1, npoints1, npoints2 // 2, 2])
    ft = tf.complex(ft[..., 0], ft[..., 1])

    ft = np.array(ft)

    ft = ng.proc_base.sp(ft, off=0.42, end=0.98, pow=2.0, inv=False, rev=False)

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


def plot_contour_wAxes(
    ax, ft_outer, h_ppm, n_ppm, col="viridis", lvl=None, invert=False
):
    ft_outer = ft_outer.numpy()[0, :, :]
    h_ppm, n_ppm = np.meshgrid(h_ppm, n_ppm)
    if lvl is None:
        lvl = getLevels(np.max(ft_outer) * 0.03, 1.2, 22)

    ax.contour(h_ppm, n_ppm, ft_outer, levels=lvl, cmap=col)
    if invert:
        ax.invert_yaxis()
        ax.invert_xaxis()


def load_ss(ssfile, max_points):
    ss = []
    with open(ssfile) as inny:
        for line in inny:
            if len(line) > 0:
                line = line.split()
                try:
                    ss.append(int(line[0]))
                    if int(line[0]) > 255:
                        print("the network can only reconstruct up to 256 complex")
                        print("points. The value ", int(line[0]), "in the sampling")
                        print("schedule is too large. Please adjust the input and")
                        print("retry. Aborting now ...")
                        sys.exit()
                except ValueError:
                    print(
                        "only integer values are permitted in the "
                        "sampling schedule (one per line)"
                    )
                    print("please check the sampling schedule for errors")
                    print("aborting now...")
                    sys.exit()

    ss = np.array(ss)
    sparsity = ss.shape[0] / max_points

    print(
        "data sparsity = ",
        sparsity * 100.0,
        "%. Sampled ",
        ss.shape[0],
        " points out of ",
        max_points,
    )

    return ss


def expand_data(data, ss, max_points, dir_points):
    exp_dat = np.zeros((max_points * 2, dir_points))
    for i in range(ss.shape[0]):
        exp_dat[2 * ss[i], :] = data[2 * i, :]
        exp_dat[2 * ss[i] + 1, :] = data[2 * i + 1, :]

    return exp_dat


def make_dl_dic(in_dic, max_points):
    dl_dic = copy.deepcopy(in_dic)
    dl_dic["FDF1TDSIZE"] = float(max_points)
    dl_dic["FDF1APOD"] = float(max_points)
    dl_dic["FDSLICECOUNT"] = float(max_points)
    dl_dic["FDSPECNUM"] = float(max_points)
    return dl_dic


def _fidnet_doRecon2D(
    model_weights, file, ss_file, max_points, outfile, f1180="y", shift="n"
):
    dic, data = ng.pipe.read(file)

    model = build_model()
    model.load_weights(model_weights)

    ss = load_ss(ss_file, max_points)

    ind_points = data.shape[0]  # sampled points in indirect dim
    dir_points = data.shape[1]  # sampled points in direct dim

    if ind_points > 512:
        print("the input spectrum contains too many sampled points")
        print("the network can have a maximum of 256 complex points in the")
        print("reconstructed spectra. Please reduce the size of the input")
        print("aborting now...")
        sys.exit()

    if ss.shape[0] == ind_points // 2:
        print(
            "number of recorded points in indirect dimension "
            "matches sampling schedule"
        )
        print("proceeding with reconstruction...")
    else:
        print(
            "there is a mismatch between the sampling "
            "schedule and number of recorded points"
        )
        print(
            "in the indirect dimension. Please check the sampling "
            "schedule or your input spectrum"
        )
        print("may need to be transposed")
        print("aborting now...")
        sys.exit()

    if max_points > 256:
        print("the maximum size of the final spectrum is 256 complex points in")
        print("the indirect dimension. The output will be truncated at this point")
        max_points = 256

    data = expand_data(data, ss, max_points, dir_points)
    data = tf.convert_to_tensor(data)

    dl_dic = make_dl_dic(dic, max_points)

    shape = tf.shape(data).numpy()
    max_val = tf.reduce_max(data)
    data = data / max_val

    Hpoints = shape[1]
    Npoints = shape[0]

    padding_2 = [[0, 512 - tf.shape(data)[0]], [0, 0]]

    data_samp = tf.pad(data, padding_2, "Constant", constant_values=0.0)

    data_samp = tf.transpose(data_samp)

    padding_recon = [[3, 3], [0, 0]]
    data_samp = tf.pad(data_samp, padding_recon, "Constant", constant_values=0.0)

    scale = np.array(
        [np.max(np.fabs(data_samp[i : i + 4, :])) for i in range(Hpoints + 3)]
    )

    sampy = np.zeros((scale.shape[0], 4, tf.shape(data_samp)[1]))
    for i in range(scale.shape[0]):
        sampy[i, :, :] = data_samp[i : i + 4, :]

    samp_av = tf.convert_to_tensor(sampy)

    samp_av = tf.transpose(samp_av, perm=[1, 2, 0])

    samp_av = samp_av / scale
    samp_av = tf.transpose(samp_av, perm=[2, 1, 0])
    samp_av = tf.expand_dims(samp_av, axis=3)

    data = tf.expand_dims(data, axis=0)

    res = model.predict(samp_av)
    res = tf.convert_to_tensor(res[0])
    res = rescale_dat(res, scale)
    res_keep = copy.deepcopy(res)
    res = get_average_results(res, Hpoints)

    res = res[:, :Npoints, :, 0]

    res_ft = ft_second(
        res, npoints1=Hpoints, npoints2=Npoints, f1180=f1180, shift=shift
    )
    data_ft = ft_second(
        data, npoints1=Hpoints, npoints2=Npoints, f1180=f1180, shift=shift
    )

    data_ft = data_ft / tf.reduce_max(data_ft)
    res_ft = res_ft / tf.reduce_max(res_ft)
    ng.pipe.write(outfile, dl_dic, res.numpy()[0], overwrite=True)

    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 3)
    ax3 = plt.subplot(2, 2, 2)
    ax4 = plt.subplot(2, 2, 4)

    plot_contour(ax1, data)
    plot_contour(ax2, data_ft, invert=True)

    plot_contour(ax3, res)
    plot_contour(ax4, res_ft, invert=True)

    plt.savefig(outfile + ".png")

    plt.close('all')

    get_ind_spectra(
        res_keep, res_ft, Hpoints, Npoints, dl_dic, outfile, f1180=f1180, shift=shift
    )
