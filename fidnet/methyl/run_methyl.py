#!/usr/bin/python

# GK September 2023
# Scripts for running methyl decoupling DNNs
#
# note in methyl_decoup_funcs model paths need to be set
# requires nmrpipe to be installed to work

import os

from fidnet.methyl import methyl_decoup_funcs


def write_initial(input, outfile, com_file, min_1H, max_1H, p0):
    with open(com_file, "w") as outy:
        outy.write("#!/bin/csh \n")
        outy.write(f"nmrPipe -in {input} \\\n")
        outy.write("| nmrPipe  -fn EM -lb 0.5 -c 0.5 \\\n")
        outy.write("| nmrPipe  -fn ZF -auto \\\n")
        outy.write("| nmrPipe  -fn FT -auto  \\\n")
        outy.write(f"| nmrPipe  -fn PS -p0 {p0} -p1 0.00 -di -verb \\\n")
        outy.write(f"| nmrPipe  -fn EXT -x1 {min_1H}ppm -xn {max_1H}ppm -sw \\\n")
        outy.write(f" -ov -out {outfile}")


def write_intermediate1(input, outfile, com_file, alt, neg, yp0, yp1, yZF):
    with open(com_file, "w") as outy:
        outy.write("#!/bin/csh \n")
        outy.write(f"nmrPipe -in {input} \\\n")
        outy.write("| nmrPipe -fn TP  \\\n")
        outy.write("| nmrPipe  -fn SP -off 0.42 -end 0.98  -pow 2 -c 0.5    \\\n")
        outy.write(f"| nmrPipe  -fn ZF -auto  -zf {yZF}\\\n")
        if alt and neg:
            outy.write("| nmrPipe  -fn FT -alt -neg  \\\n")
        elif alt:
            outy.write("| nmrPipe  -fn FT -alt \\\n")
        elif neg:
            outy.write("| nmrPipe  -fn FT -alt -neg  \\\n")
        else:
            outy.write("| nmrPipe  -fn FT -auto  \\\n")
        outy.write(f"| nmrPipe  -fn PS -p0 {yp0} -p1 {yp1} -di -verb \\\n")
        outy.write(f" -ov -out {outfile}")


def write_intermediate2(input, outfile, com_file):
    with open(com_file, "w") as outy:
        outy.write("#!/bin/csh \n")
        outy.write(f"nmrPipe -in {input} \\\n")
        outy.write("| nmrPipe -fn TP  \\\n")
        outy.write("| nmrPipe -fn HT -auto -verb 	\\\n")
        outy.write("| nmrPipe -fn FT -inv \\\n")
        outy.write("| nmrPipe -fn APOD -inv -hdr \\\n")
        outy.write("| nmrPipe -fn TP \\\n")
        outy.write(f"-ov -out {outfile}")


def write_final(input, outfile, com_file, blc=False, xZF=1):
    with open(com_file, "w") as outy:
        outy.write("#!/bin/csh \n")
        outy.write(f"nmrPipe -in {input} \\\n")
        outy.write("| nmrPipe -fn TP   \\\n")
        outy.write("| nmrPipe  -fn SP -off 0.42 -end 0.98  -pow 2 -c 0.5    \\\n")
        outy.write(f"| nmrPipe  -fn ZF -auto  -zf {xZF}\\\n")
        outy.write("| nmrPipe  -fn FT -auto      \\\n")
        outy.write("| nmrPipe  -fn PS -p0 0.00 -p1 0.00 -di -verb         \\\n")
        if blc:
            outy.write("| nmrPipe -fn POLY -auto -ord 2 \\\n")
        outy.write("| nmrPipe -fn TP       \\\n")
        if blc:
            outy.write("| nmrPipe -fn POLY -auto -ord 2 \\\n")
        outy.write(f" -ov -out {outfile}")


def run_net(
    infile,
    outfolder="dl",
    outfile="dl.ft2",
    min_1H=-1.0,
    max_1H=2.5,
    p0=0.0,
    alt=False,
    neg=False,
    yp0=0.0,
    yp1=0.0,
    blc=False,
    yZF=1,
    xZF=1,
):
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    write_initial(
        infile,
        os.path.join(outfolder, "int1.ft1"),
        os.path.join(outfolder, "int1.com"),
        min_1H,
        max_1H,
        p0,
    )
    os.system(f'/bin/csh {os.path.join(outfolder,"int1.com")}')
    methyl_decoup_funcs.do_recon_indirect(
        os.path.join(outfolder, "int1.ft1"),
        os.path.join(outfolder, "int2.ft1"),
        mode="dec",
    )
    write_intermediate1(
        os.path.join(outfolder, "int2.ft1"),
        os.path.join(outfolder, "int3.ft1"),
        os.path.join(outfolder, "int2.com"),
        alt,
        neg,
        yp0,
        yp1,
        yZF,
    )
    os.system(f'/bin/csh {os.path.join(outfolder,"int2.com")}')
    write_intermediate2(
        os.path.join(outfolder, "int3.ft1"),
        os.path.join(outfolder, "int4.ft1"),
        os.path.join(outfolder, "int3.com"),
    )
    os.system(f'/bin/csh {os.path.join(outfolder,"int3.com")}')
    methyl_decoup_funcs.do_recon_indirect(
        os.path.join(outfolder, "int4.ft1"),
        os.path.join(outfolder, "int5.ft1"),
        mode="sharp",
    )
    write_final(
        os.path.join(outfolder, "int5.ft1"),
        os.path.join(outfolder, outfile),
        os.path.join(outfolder, "fin.com"),
        blc,
        xZF,
    )
    os.system(f'/bin/csh {os.path.join(outfolder,"fin.com")}')
