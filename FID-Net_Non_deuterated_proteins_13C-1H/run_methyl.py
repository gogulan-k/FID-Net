#!/usr/bin/python

# GK September 2023
# Scripts for running methyl decoupling DNNs
#
# note in methyl_decoup_funcs model paths need to be set
# requires nmrpipe to be installed to work

import methyl_decoup_funcs
import os


def write_initial(input, outfile, com_file, min_1H, max_1H, p0):
    with open(com_file, 'w') as outy:
        outy.write('#!/bin/csh \n')
        outy.write(f'nmrPipe -in {input} \\\n')
        outy.write('| nmrPipe  -fn EM -lb 0.5 -c 0.5 \\\n')
        outy.write('| nmrPipe  -fn ZF -auto  \\\n')
        outy.write('| nmrPipe  -fn FT -auto  \\\n')
        outy.write(f'| nmrPipe  -fn PS -p0 {p0} -p1 0.00 -di -verb \\\n')
        outy.write(f'| nmrPipe  -fn EXT -x1 {min_1H}ppm -xn {max_1H}ppm -sw \\\n')
        outy.write(f' -ov -out {outfile}')

def write_intermediate1(input, outfile, com_file, alt, neg):
    with open(com_file, 'w') as outy:
        outy.write('#!/bin/csh \n')
        outy.write(f'nmrPipe -in {input} \\\n')
        outy.write('| nmrPipe -fn TP  \\\n')
        outy.write('| nmrPipe  -fn SP -off 0.42 -end 0.98  -pow 2 -c 0.5    \\\n')
        outy.write('| nmrPipe  -fn ZF -auto  \\\n')
        if alt and neg:
            outy.write('| nmrPipe  -fn FT -alt -neg  \\\n')
        elif alt:
            outy.write('| nmrPipe  -fn FT -alt \\\n')
        elif neg:
            outy.write('| nmrPipe  -fn FT -alt -neg  \\\n')
        else:
            outy.write('| nmrPipe  -fn FT -auto  \\\n')
        outy.write('| nmrPipe  -fn PS -p0 0.00 -p1 0.00 -di -verb \\\n')
        outy.write(f' -ov -out {outfile}')


def write_intermediate2(input, outfile, com_file):
    with open(com_file, 'w') as outy:
        outy.write('#!/bin/csh \n')
        outy.write(f'nmrPipe -in {input} \\\n')
        outy.write('| nmrPipe -fn TP  \\\n')
        outy.write('| nmrPipe -fn HT -auto -verb 	\\\n')
        outy.write('| nmrPipe -fn FT -inv \\\n')
        outy.write('| nmrPipe -fn APOD -inv -hdr \\\n')
        outy.write('| nmrPipe -fn TP \\\n')
        outy.write(f'-ov -out {outfile}')

def write_final(input, outfile, com_file):
    with open(com_file, 'w') as outy:
        outy.write('#!/bin/csh \n')
        outy.write(f'nmrPipe -in {input} \\\n')
        outy.write('| nmrPipe -fn TP   \\\n')
        outy.write('| nmrPipe  -fn SP -off 0.42 -end 0.98  -pow 2 -c 0.5    \\\n')
        outy.write('| nmrPipe  -fn FT -auto      \\\n')
        outy.write('| nmrPipe  -fn PS -p0 0.00 -p1 0.00 -di -verb         \\\n')
        outy.write('| nmrPipe -fn TP       \\\n')
        outy.write(f' -ov -out {outfile}')

def run_net(infile, outfolder = 'dl', outfile  ='dl.ft2', min_1H = -1.0, max_1H = 2.5, p0 = 0.0, alt = False, neg = False):
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    write_initial(infile, os.path.join(outfolder,'int1.ft1'), os.path.join(outfolder,'int1.com'), min_1H, max_1H, p0)
    os.system(f'tcsh {os.path.join(outfolder,"int1.com")}')
    methyl_decoup_funcs.do_recon_indirect(os.path.join(outfolder,'int1.ft1'),os.path.join(outfolder,'int2.ft1'), mode = 'dec')
    write_intermediate1(os.path.join(outfolder,'int2.ft1'), os.path.join(outfolder,'int3.ft1'), os.path.join(outfolder,'int2.com'), alt, neg)
    os.system(f'tcsh {os.path.join(outfolder,"int2.com")}')
    write_intermediate2(os.path.join(outfolder,'int3.ft1'), os.path.join(outfolder,'int4.ft1'), os.path.join(outfolder,'int3.com'))
    os.system(f'tcsh {os.path.join(outfolder,"int3.com")}')
    methyl_decoup_funcs.do_recon_indirect(os.path.join(outfolder,'int4.ft1'),os.path.join(outfolder,'int5.ft1'), mode = 'sharp')
    write_final(os.path.join(outfolder,'int5.ft1'), os.path.join(outfolder,outfile), os.path.join(outfolder,'fin.com'))
    os.system(f'tcsh {os.path.join(outfolder,"fin.com")}')



import argparse
parser = argparse.ArgumentParser(description='FID-Net Decouple and improve resolution of spectra for uniformly 13C-1H labelled proteins')
parser.add_argument('-in','--in', help='Input spectra. This is a 2D 13C-1H spectra (time domain data) for a uniformly labelled 13C-1H labelled protein', required=True)
parser.add_argument('-outfolder','--outfolder', help='folder where results will be saved (defaults to dl/)', required=False, default = 'dl')
parser.add_argument('-outfile','--outfile', help='filename for final processed spectrum (defaults to dl.ft2)', required=False, default = 'dl.ft2')
parser.add_argument('-min_1H','--min_1H', help='minimum 1H ppm', required=False, default = -1.0)
parser.add_argument('-max_1H','--max_1H', help='maximum 1H ppm', required=False, default = 2.5)
parser.add_argument('-p0', '--p0', help='1H phase correction', required=False, default = 0.0)
parser.add_argument('-alt','--alt', help='True/False alt flag in processing', required=False, default = False)
parser.add_argument('-neg','--neg', help='True/False neg flag in processing', required=False, default = False)

args = vars(parser.parse_args())

run_net(args['in'],args['outfolder'],args['outfile'],args['min_1H'],args['max_1H'],args['p0'],args['alt'],args['neg'])
