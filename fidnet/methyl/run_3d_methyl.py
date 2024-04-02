#!/usr/bin/python

# GK September 2023
# Scripts for running methyl decoupling DNNs
#
# note in methyl_decoup_funcs model paths need to be set
# requires nmrpipe to be installed to work

import methyl_decoup_funcs
import os

# first do nmrproc.com
# -> test.ft2 results

# then do first recon_3d
# -> dl_decoupC.ft2

# then do dl.com
# -> dl_decoupC.ft3

# then do h_sharp.com
# -> dl_decoupC.fid3

# then do second recon_3d
# -> dl_hsharp.fid3

# then do final_proc.com
# -> dl_fin.ft3

# then do rearrange.com
# -> dl_fin_rot.ft3

# then do c_sharp2.com
# -> dl_decoupC_2nd_round.fid3

# then do third recon_3d
# -> dl_csharp_2nd_round.fid3

# then do final_proc_2nd_round.com
#  -> dl_fin_2nd_round_rot.ft3

os.system(f'tcsh example_3d/nmrproc.com')
methyl_decoup_funcs.do_recon_3d('example_3d/test.ft2', 'example_3d/dl_decoupC.ft2',  mode = 'dec')
os.system(f'tcsh example_3d/dl.com')
os.system(f'tcsh example_3d/h_sharp.com')
methyl_decoup_funcs.do_recon_3d('example_3d/dl_decoupC.fid3', 'example_3d/dl_hsharp.fid3',  mode = 'sharp')
os.system(f'tcsh example_3d/final_proc.com')
os.system(f'tcsh example_3d/rearrange.com')
os.system(f'tcsh example_3d/c_sharp2.com')
methyl_decoup_funcs.do_recon_3d('example_3d/dl_decoupC_2nd_round.fid3', 'example_3d/dl_csharp_2nd_round.fid3',  mode = 'dec')
os.system(f'tcsh example_3d/final_proc_2nd_round.com')
