FID-Net: Non-Deuterated uniformly labelled Proteins
------------
------------
This code is for improving the resolution of protein spectra from uniformly 13C-1H proteins. The code requires two DNNs based on FID-Net architecture. The first network [here](https://www.dropbox.com/scl/fi/bfiec05qwf66z9b4mbqdk/fidnet_13c_methyl.h5?rlkey=mou5fx68gxazrbc9yr08j059m&dl=0) removes one 13C-13C scalar coupling and sharpens peaks in the 13C dimension. The second network [here](https://www.dropbox.com/scl/fi/z0jmv3qwnut6x5we5lf6b/fidnet_1h_methyl.h5?rlkey=zucniyig9hmg2nwgklt5mooy8&dl=0) sharpens peaks in the 1H dimension.   

Once downloaded the C13_MODEL line in methyl_decoup_funcs.py must be changed to the
path to the weights file for the 13C network (fidnet_13c_methyl.h5) and the H1_MODEL line must be changed to the path to the weights file for the 1H network (fidnet_1h_methyl.h5).

The script can then be run as follows:
python run_methyl.py -in infile

There are a number of additional options that can be accessed for processing. By typing python run_methyl.py -help
these can be accessed.

The example folder contains data for uniformly 13C-1H labelled HDAC8. The processing an be run in full using the command:

python run_methyl.py -in example/test.fid -min_1H -0.5 -max_1H 2.5 -p0 151.0 -alt True -neg True

Note nmrPipe is also required for intermediate processing steps using these scripts.

Dependencies
------------
  * [Python=3.8](https://www.python.org/downloads/)
  * [Tensorflow=2.3.1](https://www.tensorflow.org/install)
  * [NumPy=1.18.5](https://www.scipy.org/scipylib/download.html)
  * [nmrglue=0.7](https://nmrglue.readthedocs.io/en/latest/install.html)

  * [nmrPipe](https://www.ibbr.umd.edu/nmrpipe/index.html)
