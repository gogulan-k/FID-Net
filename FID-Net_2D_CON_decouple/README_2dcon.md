FID-Net: Decoupling 2D CON Spectra
------------
------------
This code is for decoupling 2D CON spectra using the FID-Net architecture.
To use the code, the file containing the weights for the trained network must be
downloaded. This is availble [here](https://www.dropbox.com/s/782hprx1onwmjbf/jcoup_2dcon.h5?dl=0).
Once downloaded the MODEL_WEIGHTS line in fidnet_2d_conDecoup.py must be changed to the
path to the weights file (fidnet_2dcon.h5).

The script can then be run as follows:
python fidnet_2d_conDecoup.py -in infile -out outfile

The out argument is optional and will default to 'decouple.ft2'

Note: the 2D CON decoupler can only work with up to 512 complex points in the
13C dimension. Spectra containing more points than this will be truncated at
512 complex points.

An example in-phase spectrum of ubiquitin (test001.ft1) is provided in the example
folder.

Dependencies
------------
  * [Python=3.8](https://www.python.org/downloads/)
  * [Tensorflow=2.3.1](https://www.tensorflow.org/install)
  * [NumPy=1.18.5](https://www.scipy.org/scipylib/download.html)
  * [nmrglue=0.7](https://nmrglue.readthedocs.io/en/latest/install.html)
  * [Matplotlib=3.2.2](http://matplotlib.org/users/installing.html)

  The FID-Net 2D CON decoupling script has been written and tested with the
  above dependencies. Performance with other module versions has not been tested.


Preparing Data
-------------
The input to the DNN must be in nmrPipe format. If using a Bruker spectrometer,
the raw FID file (ser) must be converted to nmrpipe format using the DMX flag for
FID-Net decoupling to perform correctly.  

Prior to input into FID-Net, the direct dimension of the spectrum is phased but
the imaginary part is not deleted. The spectrum is then transposed, apodized,
zero-filled, phased and Fourier transformed in the indirect dimension. For best
results excessive zero-filling in the indirect dimension should be avoided.
Typically we would just use 'ZF -auto' in nmrPipe. The spectrum should then be
transposed before entry into FID-Net.

The input to the DNN must be a 2D in-phase interferogram (ie. processed in the
indirect dimension but not the direct dimension as described above). If the data
is 3D it must be converted to a set of 2D planes using the pipe2xyz utility or
similar. An exemplar input is provided in the example folder (test001.ft1).

The output of the DNN can then be processed (apodized, zero-filled, Fourier
transformed and the imaginary part deleted) to give the final result. An example
(final_proc.com) is provided in the example folder.
