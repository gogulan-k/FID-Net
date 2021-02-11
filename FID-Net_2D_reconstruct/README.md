FID-Net: Reconstructing 2D NUS Spectra
------------
------------
This code is for reconstructing 2D NUS NMR spectra using the FID-Net architecture.
To use the code, the file containing the weights for the trained network must be
downloaded. This is availble [here](https://www.dropbox.com/s/6qfaoae7n96mfuj/fidnet_recon.h5?dl=0).
Once downloaded the MODEL_WEIGHTS line in fidnet_recon.py must be changed to the
path to the weights file (fidnet_recon.h5). 

The script can then be run as follows:
python fidnet_recon.py -in infile -ss sampling_sched -max max_num_complex_points

e.g. for the example spectrum:
python fidnet_recon.py -in example/hdac_ex.ft1 -ss example/ss_hdac.out -max 192

Dependencies
------------
  * [Python=3.8](https://www.python.org/downloads/)
  * [Tensorflow=2.3.1](https://www.tensorflow.org/install)
  * [NumPy=1.18.5](https://www.scipy.org/scipylib/download.html)
  * [Matplotlib=3.2.2](http://matplotlib.org/users/installing.html)
  * [nmrglue=0.7](https://nmrglue.readthedocs.io/en/latest/install.html)
  
The FID-Net reconstruction script has been written and tested with the
above dependencies. Performance with other module versions has not been tested.
