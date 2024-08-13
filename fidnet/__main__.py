from pathlib import Path

import typer
from click import Context
from rich import print
from typer.core import TyperGroup

from fidnet import config


class OrderCommands(TyperGroup):
    def list_commands(self, ctx: Context):
        """Return list of commands in the order
        they should appear."""
        return list(self.commands)  # get commands using self.commands


cli = typer.Typer(
    cls=OrderCommands,
    name="fidnet",
    epilog="Thanks!",
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="""
Deep Neural Networks for Analysing NMR time domain data.

https://github.com/gogulan-k/FID-Net

""",
)


@cli.command("ca_detect", no_args_is_help=True)
def ca_detect(
    infile: Path = typer.Option(
        help="Help text in the original was " "the same as for con_decouple"
    ),
    outfile: Path = typer.Option(
        default="fidnet_ca_detect.ft1", help="Path to the output file."
    ),
):
    """FID-Net 2D CA detect"""

    # from fidnet.ca_detect.fidnet_2d_caDetect import direct_decouple
    from fidnet.experiments import run_ca_direct_decouple

    if str(infile) == "example":
        infile = config.example_file_ca_detect

    run_ca_direct_decouple(infile, outfile)


@cli.command("con_decouple", no_args_is_help=True)
def con_decouple(
    infile: Path = typer.Option(
        help="Input spectra. This is a 2D in phase CON "
        "spectra with the 13C dimension in the time "
        "domain. The 13C dimension should be phased "
        "but the imaginary part retained and should "
        "not be apodized, zero-filled or Fourier "
        "transformed. The 15N dimension should be "
        "apodized, zero-filled, phased (the imaginary "
        "part deleted) then Fourier transformed. "
        "The order of the input dimensions must "
        "be 15N, 13C."
    ),
    outfile: Path = typer.Option(
        default="fidnet_con_decoupled.ft1", help="Path to the output file."
    ),
):
    """FID-Net 2D CON decoupling"""

    # from fidnet.con_decouple.fidnet_2d_conDecoup import direct_decouple
    from fidnet.experiments import run_con_direct_decouple

    if str(infile) == "example":
        infile = config.example_file_con_decouple

    run_con_direct_decouple(infile, outfile)


@cli.command("ctcp_decouple", no_args_is_help=True)
def ctcp_decouple(
    infile: Path = typer.Option(
        help="Input  spectra. This is a 2D in-phase Ct-Cp "
        "spectra with the 13Ct dimension in the time "
        "domain. The 13Ct dimension should be phased "
        "but the imaginary part retained and should not "
        "be apodized, zero-filled or Fourier transformed. "
        "The 13Cp dimension should be apodized, "
        "zero-filled, phased (the imaginary part deleted) "
        "then Fourier transformed. The order of the input "
        "dimensions must be 13Cp, 13Ct."
    ),
    outfile: Path = typer.Option(
        default="fidnet_ctcp_decoupled.ft1", help="Path to the output file."
    ),
):
    """FID-Net 2D CTCP decoupling"""

    # from fidnet.ctcp_decouple.fidnet_2d_ctcpDecoup import direct_decouple
    from fidnet.experiments import run_ctcp_direct_decouple

    if str(infile) == "example":
        infile = config.example_file_ctcp

    run_ctcp_direct_decouple(infile, outfile)


@cli.command("methyl", no_args_is_help=True)
def methyl(
    infile: Path = typer.Option(
        help="Input spectra. This is a 2D 13C-1H"
        "spectra (time domain data) for"
        "a uniformly labelled 13C-1H labelled protein."
        "If using literally 'example',"
        "an example file is used"
    ),
    outdir: Path = typer.Option(
        default="fidnet_out", help="folder where results" "will be saved."
    ),
    outfile: Path = typer.Option(
        default="fidnet_methyl_CH.ft2", help="filename for final" "processed spectrum."
    ),
    min_1H: float = typer.Option(default=-1.0, help="minimum 1H ppm"),
    max_1H: float = typer.Option(default=2.5, help="maximum 1H ppm"),
    p0: float = typer.Option(default=0.0, help="1H phase correction"),
    alt: bool = typer.Option(
        default=False, help="NMRPipe: dimension is left/right swapped"
    ),
    neg: bool = typer.Option(default=False, help="NMRPipe: dimension is reversed"),
):
    """FID-Net Decouple and improve resolution
    of spectra for uniformly 13C-1H labelled
    proteins.
    """
    from fidnet.experiments import run_methyl

    if str(infile) == "example":
        infile = config.example_file_non_deuterated

    run_methyl(
        infile=infile,
        outdir=outdir,
        outfile=outfile,
        min_1H=min_1H,
        max_1H=max_1H,
        p0=p0,
        alt=alt,
        neg=neg,
    )


@cli.command("hnca", no_args_is_help=True)
def hnca(
    infile: Path = typer.Option(
        help="Input  spectra. This is a 3D HNCA or"
        "HN(CO)CA spectra with the 13C dimension in the time domain."
        "The 15N and 1H dimensions should be phased and Fourier transformed."
        "The order of the input dimensions must be 1H,15N, 13C."
    ),
    outfile: Path = typer.Option(default="fidnet_hnca_decoupled.ft2", help="out file"),
):
    """FID-Net 3D HNCA decoupling."""

    # from fidnet.hnca.fidnet_3d_decouple import decouple_spec
    from fidnet.experiments import run_3d_hnca

    if str(infile) == "example":
        infile = config.example_file_hnca

    run_3d_hnca(infile, outfile)


@cli.command("aromatic", no_args_is_help=True)
def aromatic(
    infile: Path = typer.Option(
        help = "this is the input spectrum (pseudo-3D-nmrPipe-file.ft1')"
    ),
        outfile: Path = typer.Option(
        default="aromatic_output.ft2", help="name of the output file"
    ),
    UseGPU: bool = typer.Option(
        default = True,
        help = "True to use GPU, False for CPU analysis"
    ),
    GPUIDX: int = typer.Option(
        default = None,
        help = "GPU number to use"
    ),
    offset1h: float = typer.Option(
        default = 0.4,
        help = "Set the offset for the sine-squared window function "
                "in the 1H dimension. Default is 0.40, which was "
                "used during training"
    ),
    offset13c: float = typer.Option(
        default = 0.4,
        help = "Set the offset for the sine-squared window function "
                "in the 13C dimension. Default is 0.40, which was "
                "used during training"        
    )
):
    from fidnet.experiments import run_aromatic
    run_aromatic(infile, outfile, UseGPU, GPUIDX, offset1h, offset13c)

@cli.command("reconstruct", no_args_is_help=True)
def reconstruct(
    infile: Path = typer.Option(
        help="this is the measured 2D non-uniformly sampled spectra. It should "
        "be processed in the direct dimension, phased and transposed. The "
        "indirect dimension should not be processed in any way. The unsampled "
        "points in the indirect dimension should not be replaced with zeros "
        "(for example by using the nusExpand tool) this is taken care of by "
        "the program itself. The maximum number of complex points in the "
        "indirect dimension that can be included in the network is 256. The"
        "spectrum will be truncated after this."
    ),
    sampling_schedule: Path = typer.Option(
        help="this is the sampling schedule used. This is simply a list of"
        "integers (oneinteger per line) giving the complex points that "
        "are measured in the NUS experiment."
    ),
    max_points: int = typer.Option(
        help="this is the number of complex points in the final output. I.e."
        "the sparsity is given by the number of values in the sampling"
        "schedule divided by this value."
    ),
    outfile: Path = typer.Option(
        default="fidnet_nus_reconstructed.ft1", help="name of the output file"
    ),
    f1180: bool = typer.Option(
        default=True,
        help="f1180 flag (y/n) only important for matplotlib output "
        "and fidnet_reconstructed.ft2",
    ),
    shift: bool = typer.Option(
        default=False,
        help="frequency shift flag (y/n) only important for "
        "matplotlib output and std.ft2",
    ),
):
    """FID-Net 2D Non-Uniform Sampling (NUS) reconstruction"""

    # from fidnet.nus.fidnet_recon import _fidnet_doRecon2D
    from fidnet.experiments import run_nus_reconstruction

    if str(infile) == "example":
        infile = config.example_file_nus_reconstruct
        sampling_schedule = config.example_file_nus_sampling_schedule

    run_nus_reconstruction(
        infile,
        sampling_schedule,
        max_points,
        outfile,
        f1180=f1180,
        shift=shift,
    )


@cli.command("run-examples")
def run_examples(skip_3d: bool = typer.Option(True, help="Skip 3D example")):
    """Run all the examples in one go."""
    from fidnet.experiments import run_examples
    from fidnet.util import download_example_data

    download_example_data()
    run_examples(skip_3d=skip_3d)


@cli.command("download-example-data")
def download_example_data(force: bool = False):
    """Download example data to try out the different
    FID-Net functions.
    """
    from fidnet.util import download_example_data

    download_example_data(force=force)


@cli.command("download-weights")
def download_all_weights(force: bool = False):
    """
    Download the weights for all FID-Net models. Running
    this is not strictly necessary as the weights are
    downloaded on the fly for individual models when
    they are not present yet.
    """
    from fidnet.util import download_all_weights

    download_all_weights(force=force)


@cli.command("settings")
def settings():
    text = str(config)

    text += """

    To change some of these settings (for example FIDNET_WEIGHTS_DIR)
    either:

    1) Set an environment value manually (potentially just before
        running a command):

        FIDNET_WEIGHTS_DIR=/some/custom/path fidnet [COMMAND]

    2) Add a line to a .env file:

        FIDNET_WEIGHTS_DIR=/some/custom/path

    For more information about settings management read:
    https://pydantic-docs.helpmanual.io/usage/settings/

    """

    print(text)


@cli.command("version")
def show_version():
    """
    Show the version of the nucleotides library.
    """

    from fidnet._version import version

    print(version)


if __name__ == "__main__":
    cli()
