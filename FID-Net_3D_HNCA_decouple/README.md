FID-Net: Decoupling 3D HNCA Spectra
------------
------------
This code is for decoupling 3D HNCA and HN(COCA) spectra using the FID-Net architecture.
To use the code, the file containing the weights for the trained network must be
downloaded. This is availble [here](https://www.dropbox.com/s/v4bw5hgst2q3hwi/fidnet_3dca_decouple.h5?dl=0).
Once downloaded the MODEL_WEIGHTS line in fidnet_recon.py must be changed to the
path to the weights file (fidnet_3dca_decouple.h5).

The script can then be run as follows:
python fidnet_3d_decouple.py -in infile -out outfile

The out argument is optional and will default to 'decouple.ft2'

Dependencies
------------
  * [Python=3.8](https://www.python.org/downloads/)
  * [Tensorflow=2.3.1](https://www.tensorflow.org/install)
  * [NumPy=1.18.5](https://www.scipy.org/scipylib/download.html)
  * [nmrglue=0.7](https://nmrglue.readthedocs.io/en/latest/install.html)

  The FID-Net 3D decoupling script has been written and tested with the
  above dependencies. Performance with other module versions has not been tested.
