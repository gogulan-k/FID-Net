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

-in : this is the measured 2D non-uniformly sampled spectra. It should be processed
      in the direct dimension, phased and transposed. The indirect dimension
      should not be processed in any way. The unsampled points in the indirect
      dimension should not be replaced with zeros (for example by using the
      nusExpand tool) this is taken care of by the program itself. The maximum
      number of complex points in the indirect dimension that can be included
      in the network is 256. The spectrum will be truncated after this.

-ss : this is the sampling schedule used. This is simply a list of integers (one
      integer per line) giving the complex points that are measured in the NUS
      experiment.

-max: this is the number of complex points in the final output. I.e. the sparsity
      is given by the number of values in the sampling schedule divided by this
      value.

The output of the network is an nmrPipe file with the indirect dimension reconstructed
in the time domain. The indirect dimension can now be processed (apodized, zero-filled,
phased and Fourier transformed) to yield the final reconstructed spectrum. The
analysis also outputs std.ft2, providing a measure of confidence in the outputs.
This is also in nmrPipe format and is pre-processed and Fourier transformed according
to default parameters. If these are incorrect a Hilbert transform and inverse Fourier
transform can be applied to put this back into the time domain before reprocessing. 

Dependencies
------------
  * [Python=3.8](https://www.python.org/downloads/)
  * [Tensorflow=2.3.1](https://www.tensorflow.org/install)
  * [NumPy=1.18.5](https://www.scipy.org/scipylib/download.html)
  * [Matplotlib=3.2.2](http://matplotlib.org/users/installing.html)
  * [nmrglue=0.7](https://nmrglue.readthedocs.io/en/latest/install.html)

The FID-Net reconstruction script has been written and tested with the
above dependencies. Performance with other module versions has not been tested.
