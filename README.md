# FID-Net
Deep Neural Networks for Analysing NMR time domain data. See:

* [G Karunanithy and D F Hansen (2021, JBNMR)](https://doi.org/10.1007/s10858-021-00366-w)
* [G Karunanithy, H W Mackenzie and D F Hansen (2021, JACS)](https://doi.org/10.1021/jacs.1c04010)
* [G Karunanithy, V K Shukla and D F Hansen (2024, Nature Communications)](https://doi.org/10.1038/s41467-024-49378-8)
* [V K Shukla, G Karunanithy, P Vallurupalli, D F Hansen (2024, bioRxiv)](https://doi.org/10.1101/2024.04.01.587635)

## Table of Contents
- [Quick Start](#quick-start)
- [Installation](#installation)
    - [Conda/Mamba environment](#conda-mamba-environment)
    - [Weights of the Neural Networks](#weights-of-the-neural-networks)
    - [Verify whether things are working on example data](#verify-whether-things-are-working-on-example-data)
- [Usage](#usage)
- [README's for the individual experiments](#readmes-for-the-individual-experiments)
    - [FID-Net 2D CA detect](#fid-net-2d-ca-detect)
    - [FID-Net 2D CON decoupling](#fid-net-2d-con-decoupling)
    - [FID-Net 2D CTCP decoupling](#fid-net-2d-ctcp-decoupling)
    - [FID-Net 2D NUS reconstruction](#fid-net-2d-nus-reconstruction)
    - [3D HNCA Decoupling](#3d-hnca-decoupling)
    - [Methyl Decoupling](#methyl-decoupling)
    - [FID-Net2 for Aromatic Sidechains](#FID-Net2-for-Aromatic-Sidechains)
- [Development](#development)

## Quick Start

This is the short version of this README. For more details, see more
detailed explanations below.

```shell
git clone https://github.com/gogulan-k/FID-Net.git
cd FID-Net
# Following two command only if you don't have NMRPipe installed:
chmod +x install_nmrpipe.sh
install_nmrpipe.sh
mamba env update -f environment.yml
mamba activate fidnet
fidnet run-examples
```

## Installation

### Conda/Mamba environment
First, clone the repository:

```shell
git clone https://github.com/gogulan-k/FID-Net.git
```

The easiest way get a working environment with all packages that FID-Net needs,
use [conda](https://conda.io) or [mamba](https://mamba.readthedocs.io/en/latest/)
and the provided environment.yml file:

```shell
cd FID-Net
mamba env update -f environment.yml
```

and activate the environment:

```shell
mamba activate fidnet
```

Installing the environment also installs the "fidnet" package,
making the **fidnet** command line tool available (see below).


### Weights of the Neural Networks
The weights of the neural networks are not included in this python
package, but will be downloaded on the fly when needed.

If you want to manually trigger downloading the weights for all
different models at once, type:

```shell
fidnet download-weights
```

The weights are downloaded by default to the
gitignored directory:

```
<REPOSITORY ROOT>/data/weights
```

You can change settings like these by adding a .env file
or setting environment variables specifying the
FIDNET_DATA_DIR or FIDNET_WEIGHTS_DIR:

```text
# .env
FIDNET_WEIGHTS_DIR=/path/to/directory/with/weights.hd5
```

To take a look at all such settings type:

```shell
fidnet settings
```

### Verify whether things are working on example data
If you have a working environment (and NMRPipe installed, if not
see next section), you can test whether things are working
by running all examples at once:

```shell
fidnet run-examples
```

This will download example data, run all the different FID-Net functions
(except the 3D HNCA decoupler, which takes a lot longer to run).
If you just want to download the example data, without doing the processing
by the models:

```shell
fidnet download-example-data
```

### NMRPipe
NMRPipe can not be installed using conda. If you don't have it installed
yet, you can use the provided script to install it.

```shell
chmod +x install_nmrpipe.sh
install_nmrpipe.sh
```

NMRPipe gives some instructions about how to edit your .cshrc. It will
look  something like this:

```shell
 if (-e <REPO_DIR>/bin/NMRPipe/com/nmrInit.linux212_64.com) then
    source <REPO_DIR>/bin/NMRPipe/com/nmrInit.linux212_64.com
 endif
```

Follow those instructions, so that the NMRPipe command can be found.

## Usage

Please refer to the **--help** in the command line tool. Each individual command has
its own help, explaining what the input arguments are.

```shell
(fidnet) ➜  ~ fidnet --help

 Usage: fidnet [OPTIONS] COMMAND [ARGS]...

Deep Neural Networks for Analysing NMR time domain data.

https://github.com/gogulan-k/FID-Net

╭─ Options ───────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.     │
│ --show-completion             Show completion for the current shell, to     │
│                               copy it or customize the installation.        │
│ --help                        Show this message and exit.                   │
╰─────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────╮
│ ca_detect              FID-Net 2D CA detect                                 │
│ con_decouple           FID-Net 2D CON decoupling                            │
│ ctcp_decouple          FID-Net 2D CTCP decoupling                           │
│ methyl                 FID-Net Decouple and improve resolution              │
│                        of spectra for uniformly 13C-1H labelled             │
│                        proteins.                                            │
│ hnca                   FID-Net 3D HNCA decoupling.                          │
│ reconstruct            FID-Net 2D Non-Uniform Sampling (NUS) reconstruction |
| aromatic               FID-Net2 for spectra for Aromatic Sidechains         │
│ run-examples           Run all the examples in one go.                      │
│ download-example-data  Download example data to try out the different       │
│                        FID-Net functions.                                   │
│ download-weights       Download the weights for all FID-Net models. Running │
│                        this is not strictly necessary as the weights are    │
│                        downloaded on the fly for individual models when     │
│                        they are not present yet.                            │
│ settings                                                                    │
│ version                Show the version of the nucleotides library.         │
╰─────────────────────────────────────────────────────────────────────────────╯

 Thanks!
```

## README's for the individual experiments

### FID-Net 2D CA detect

```shell
(fidnet) ➜  ~ fidnet ca_detect --help

 Usage: fidnet ca_detect [OPTIONS]

 FID-Net 2D CA detect

╭─ Options ───────────────────────────────────────────────────────────────────────╮
│ *  --infile         PATH  Help text in the original was the same as for         │
│                           con_decouple                                          │
│                           [default: None]                                       │
│                           [required]                                            │
│    --outfile        PATH  Path to the output file.                              │
│                           [default: fidnet_ca_detect.ft1]                       │
│    --help                 Show this message and exit.                           │
╰─────────────────────────────────────────────────────────────────────────────────╯
```

Note: the decoupler can only work with up to 512 complex points in the
13C dimension. Spectra containing more points than this will be truncated at
512 complex points.

An example antiphase, in-phase (AP-IP) spectrum of **T4L99A** (test.ft1) is
provided in the example folder.

#### Preparing Data
The input to the DNN must be in nmrPipe format. If using a Bruker spectrometer,
the raw FID file (ser) must be converted to nmrpipe format using the DMX flag for
FID-Net decoupling to perform correctly.

Prior to input into FID-Net, the direct dimension of the spectrum is phased but
the imaginary part is not deleted. The spectrum is then transposed, apodized,
zero-filled, phased and Fourier transformed in the indirect dimension. For best
results excessive zero-filling in the indirect dimension should be avoided.
Typically we would just use 'ZF -auto' in nmrPipe. The spectrum should then be
transposed before entry into FID-Net.

The input to the DNN must be a 2D in-phase interferogram (i.e. processed in the
indirect dimension but not the direct dimension as described above).

The output of the DNN can then be processed (apodized, zero-filled, Fourier
transformed and the imaginary part deleted) to give the final result. An example
(final_proc.com) is provided in the example folder.

### FID-Net 2D CON decoupling

```shell
(fidnet) ➜  ~ fidnet con_decouple --help

 Usage: fidnet con_decouple [OPTIONS]

 FID-Net 2D CON decoupling

╭─ Options ───────────────────────────────────────────────────────────────────────╮
│ *  --infile         PATH  Input spectra. This is a 2D in phase CON spectra with │
│                           the 13C dimension in the time domain. The 13C         │
│                           dimension should be phased but the imaginary part     │
│                           retained and should not be apodized, zero-filled or   │
│                           Fourier transformed. The 15N dimension should be      │
│                           apodized, zero-filled, phased (the imaginary part     │
│                           deleted) then Fourier transformed. The order of the   │
│                           input dimensions must be 15N, 13C.                    │
│                           [default: None]                                       │
│                           [required]                                            │
│    --outfile        PATH  Path to the output file.                              │
│                           [default: fidnet_con_decoupled.ft1]                   │
│    --help                 Show this message and exit.                           │
╰─────────────────────────────────────────────────────────────────────────────────╯
```

Note: the 2D CON decoupler can only work with up to 512 complex points in the
13C dimension. Spectra containing more points than this will be truncated at
512 complex points.

An example in-phase spectrum of **ubiquitin** (test001.ft1) is provided in the example
folder.

#### Preparing Data
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

### FID-Net 2D CTCP decoupling

```shell
(fidnet) ➜  ~ fidnet ctcp_decouple --help

 Usage: fidnet ctcp_decouple [OPTIONS]

 FID-Net 2D CTCP decoupling

╭─ Options ───────────────────────────────────────────────────────────────────────╮
│ *  --infile         PATH  Input  spectra. This is a 2D in-phase Ct-Cp spectra   │
│                           with the 13Ct dimension in the time domain. The 13Ct  │
│                           dimension should be phased but the imaginary part     │
│                           retained and should not be apodized, zero-filled or   │
│                           Fourier transformed. The 13Cp dimension should be     │
│                           apodized, zero-filled, phased (the imaginary part     │
│                           deleted) then Fourier transformed. The order of the   │
│                           input dimensions must be 13Cp, 13Ct.                  │
│                           [default: None]                                       │
│                           [required]                                            │
│    --outfile        PATH  Path to the output file.                              │
│                           [default: fidnet_ctcp_decoupled.ft1]                  │
│    --help                 Show this message and exit.                           │
╰─────────────────────────────────────────────────────────────────────────────────╯
```

Note: the 2D 13Ct-13Cp decoupler can only work with up to 512 complex points in the
13C dimension. Spectra containing more points than this will be truncated at
512 complex points.

An example in-phase spectrum of **ubiquitin** (test001.ft1) is provided in the example
folder.

#### Preparing Data
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

### FID-Net 2D NUS reconstruction

```shell
(fidnet) ➜  ~ fidnet reconstruct --help

 Usage: fidnet reconstruct [OPTIONS]

 FID-Net 2D Non-Uniform Sampling (NUS) reconstruction

╭─ Options ───────────────────────────────────────────────────────────────────────╮
│ *  --infile                             PATH     this is the measured 2D        │
│                                                  non-uniformly sampled spectra. │
│                                                  It should be processed in the  │
│                                                  direct dimension, phased and   │
│                                                  transposed. The indirect       │
│                                                  dimension should not be        │
│                                                  processed in any way. The      │
│                                                  unsampled points in the        │
│                                                  indirect dimension should not  │
│                                                  be replaced with zeros (for    │
│                                                  example by using the nusExpand │
│                                                  tool) this is taken care of by │
│                                                  the program itself. The        │
│                                                  maximum number of complex      │
│                                                  points in the indirect         │
│                                                  dimension that can be included │
│                                                  in the network is 256.         │
│                                                  Thespectrum will be truncated  │
│                                                  after this.                    │
│                                                  [default: None]                │
│                                                  [required]                     │
│ *  --sampling-schedule                  PATH     this is the sampling schedule  │
│                                                  used. This is simply a list    │
│                                                  ofintegers (oneinteger per     │
│                                                  line) giving the complex       │
│                                                  points that are measured in    │
│                                                  the NUS experiment.            │
│                                                  [default: None]                │
│                                                  [required]                     │
│ *  --max-points                         INTEGER  this is the number of complex  │
│                                                  points in the final output.    │
│                                                  I.e.the sparsity is given by   │
│                                                  the number of values in the    │
│                                                  samplingschedule divided by    │
│                                                  this value.                    │
│                                                  [default: None]                │
│                                                  [required]                     │
│    --outfile                            PATH     name of the output file        │
│                                                  [default:                      │
│                                                  fidnet_nus_reconstructed.ft1]  │
│    --f1180                --no-f1180             f1180 flag (y/n) only          │
│                                                  important for matplotlib       │
│                                                  output and                     │
│                                                  fidnet_reconstructed.ft2       │
│                                                  [default: f1180]               │
│    --shift                --no-shift             frequency shift flag (y/n)     │
│                                                  only important for matplotlib  │
│                                                  output and std.ft2             │
│                                                  [default: no-shift]            │
│    --help                                        Show this message and exit.    │
╰─────────────────────────────────────────────────────────────────────────────────╯
```

This code is for reconstructing 2D NUS NMR spectra using the FID-Net architecture.
To use the code, the file containing the weights for the trained network must be
downloaded.

The output of the network is an nmrPipe file with the indirect dimension reconstructed
in the time domain. The indirect dimension can now be processed (apodized, zero-filled,
phased and Fourier transformed) to yield the final reconstructed spectrum. The
analysis also outputs std.ft2, providing a measure of confidence in the outputs.
This is also in nmrPipe format and is pre-processed and Fourier transformed according
to default parameters. If these are incorrect a Hilbert transform and inverse Fourier
transform can be applied to put this back into the time domain before reprocessing.

There is an example file for HDAC in the example folder, together with the sampling
schedule.

### 3D HNCA Decoupling

```shell
(fidnet) ➜  ~ fidnet hnca --help

 Usage: fidnet hnca [OPTIONS]

 FID-Net 3D HNCA decoupling.

╭─ Options ───────────────────────────────────────────────────────────────────────╮
│ *  --infile         PATH  Input  spectra. This is a 3D HNCA orHN(CO)CA spectra  │
│                           with the 13C dimension in the time domain.The 15N and │
│                           1H dimensions should be phased and Fourier            │
│                           transformed.The order of the input dimensions must be │
│                           1H,15N, 13C.                                          │
│                           [default: None]                                       │
│                           [required]                                            │
│    --outfile        PATH  out file [default: fidnet_hnca_decoupled.ft2]         │
│    --help                 Show this message and exit.                           │
╰─────────────────────────────────────────────────────────────────────────────────╯
```

This code is for decoupling 3D HNCA and HN(COCA) spectra using the FID-Net
architecture. Note: the 3D HNCA decoupler can only work with up to 256 complex
points in the 13C dimension. Spectra containing more points than this will be
truncated at 256 complex points.

### Methyl Decoupling

```shell
(fidnet) ➜  ~ fidnet methyl --help

 Usage: fidnet methyl [OPTIONS]

 FID-Net Decouple and improve resolution of spectra for uniformly 13C-1H labelled
 proteins.

╭─ Options ───────────────────────────────────────────────────────────────────────╮
│ *  --infile                 PATH   Input spectra. This is a 2D 13C-1Hspectra    │
│                                    (time domain data) fora uniformly labelled   │
│                                    13C-1H labelled protein.If using literally   │
│                                    'example',an example file is used            │
│                                    [default: None]                              │
│                                    [required]                                   │
│    --outdir                 PATH   folder where resultswill be saved.           │
│                                    [default: fidnet_out]                        │
│    --outfile                PATH   filename for finalprocessed spectrum.        │
│                                    [default: fidnet_methyl_CH.ft2]              │
│    --min-1h                 FLOAT  minimum 1H ppm [default: -1.0]               │
│    --max-1h                 FLOAT  maximum 1H ppm [default: 2.5]                │
│    --p0                     FLOAT  1H phase correction [default: 0.0]           │
│    --alt        --no-alt           NMRPipe: dimension is left/right swapped     │
│                                    [default: no-alt]                            │
│    --neg        --no-neg           NMRPipe: dimension is reversed               │
│                                    [default: no-neg]                            │
│    --help                          Show this message and exit.                  │
╰─────────────────────────────────────────────────────────────────────────────────╯
```

This code is for improving the resolution of protein spectra from uniformly 13C-1H
proteins. The code requires two DNNs based on FID-Net architecture. The first
network removes one 13C-13C scalar coupling and sharpens peaks in the 13C dimension.
The second network sharpens peaks in the 1H dimension.

The example folder contains data for uniformly 13C-1H labelled HDAC8.


### FID-Net2 for Aromatic Sidechains

```shell
(fidnet) ➜  ~ fidnet aromatic --help

 Usage: fidnet aromatic [OPTIONS]

 FID-Net2 ransforms NMR spectra recorded on simple uniformly 13C labelled samples to 
 yield high-quality 1H-13C correlation spectra of the aromatic side chains. 
 Spectra should be recorded with the dedicated pulse programme

╭─ Options ───────────────────────────────────────────────────────────────────────╮
│ *  --infile                 PATH   Input spectra. This should be a pseudo-3D    │
│                                    NMR pipe file that has been recorded using   │
│                                    the dedicated pulse sequence (see folder)    │
│                                                                                 │
│                                    [default: None]                              │
│                                    [required]                                   │
│    --outfile                PATH   filename for final processed spectrum.       │
│                                    [default: aromatic_output.ft2]               │
│    --UseGPU                 BOOL   True to use GPU.                             │
|                                    [default: True]                              |
│    --GPUIDX                 INT    GPU number to use                            │
|                                    [default: None]                              |
│    --offset1h               FLOAT  Set the offset for the sine-squared window   |
|                                    function in the 1H dimension. Default is     |
|                                    0.40, which was used during training         │
│                                    [default: 0.4]                               |
│    --offset13c              FLOAT  Set the offset for the sine-squared window   |
|                                    function in the 1H dimension. Default is     |
|                                    0.40, which was used during training         │
│                                    [default: 0.4]                               |
│    --help                          Show this message and exit.                  │
╰─────────────────────────────────────────────────────────────────────────────────╯
```

## Development
You can install pre-commit hooks that do some checks before you commit your code:

```
pip install -e ".[dev]"
pre-commit install
```
