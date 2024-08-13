from pathlib import Path

from fidnet import config
from fidnet.ca_detect.fidnet_2d_caDetect import direct_decouple as _ca_direct_decouple
from fidnet.con_decouple.fidnet_2d_conDecoup import direct_decouple as _con_decouple
from fidnet.ctcp_decouple.fidnet_2d_ctcpDecoup import direct_decouple as _ctcp_decouple
from fidnet.hnca.fidnet_3d_decouple import decouple_spec as _hnca_decouple_spec
from fidnet.methyl.run_methyl import run_net as _methyl_run_net
from fidnet.nus.fidnet_recon import _fidnet_doRecon2D
from fidnet.aromatic_fidnet2.aromatic_fidnet2 import _aromatic_fidnet2
from fidnet.util import download_weights


def run_ca_direct_decouple(infile: Path, outfile: Path = Path("fidnet_ca_detect.ft1")):
    weights = config.weights_ca_detect
    download_weights(weights)
    _ca_direct_decouple(weights, infile, str(outfile))


def run_con_direct_decouple(
    infile: Path, outfile: Path = Path("fidnet_con_decoupled.ft1")
):
    weights = config.weights_con_decouple
    download_weights(weights)
    _con_decouple(weights, infile, str(outfile))


def run_ctcp_direct_decouple(
    infile: Path, outfile: Path = Path("fidnet_ctcp_decoupled.ft1")
):
    weights = config.weights_ctcp
    download_weights(weights)
    _ctcp_decouple(weights, infile, str(outfile))


def run_nus_reconstruction(
    infile: Path,
    sampling_schedule: Path,
    max_points: int,
    outfile: Path = Path("fidnet_nus_reconstructed.ft1"),
    f1180: bool = True,
    shift: bool = False,
):
    weights = config.weights_nus_reconstruct
    download_weights(weights)
    _fidnet_doRecon2D(
        weights,
        infile,
        sampling_schedule,
        max_points,
        str(outfile),
        f1180=f1180,
        shift=shift,
    )


def run_aromatic(
        infile: Path,
        outfile: Path,
        UseGPU: bool = True,
        GPUIDX: int = None,
        offset1h: float = 0.4,
        offset13c: float = 0.4,
):
    weights = config.weights_aromatic
    download_weights(weights)
    _aromatic_fidnet2(
        infile,
        outfile,
        weights,
        UseGPU,
        GPUIDX,
        offset1h,
        offset13c,
    )
    

def run_methyl(
    infile: Path,
    outdir: Path = Path("fidnet_out"),
    outfile: Path = Path("fidnet_methyl_CH.ft2"),
    min_1H: float = -1.0,
    max_1H: float = 2.5,
    p0: float = 0.0,
    alt: bool = False,
    neg: bool = False,
):
    download_weights(config.weights_1h_methyl)
    download_weights(config.weights_13c_methyl)
    _methyl_run_net(
        infile=infile,
        outfolder=outdir,
        outfile=outfile,
        min_1H=min_1H,
        max_1H=max_1H,
        p0=p0,
        alt=alt,
        neg=neg,
    )


def run_3d_hnca(infile: Path, outfile: Path = Path("fidnet_hnca_decoupled.ft2")):
    weights = config.weights_3D_HNCA_decouple
    download_weights(weights)
    _hnca_decouple_spec(
        weights,
        infile,
        str(outfile),
    )


def run_examples(skip_3d: bool = True):
    """Run all the examples in one go."""
    out_dir = Path("example_out")
    print("\n1/7: Running CA detect example.")
    run_ca_direct_decouple(config.example_file_ca_detect, out_dir / "ca.ft1")

    print("\n2/7: Running CON decouple example.")
    run_con_direct_decouple(config.example_file_con_decouple, out_dir / "con_decouple.ft1")

    print("\n3/7: Running CTCP decouple example.")
    run_ctcp_direct_decouple(config.example_file_ctcp, out_dir / "ctcp.ft1")

    print("\n4/7: Running NUS reconstruction example.")
    run_nus_reconstruction(
        infile=config.example_file_nus_reconstruct,
        sampling_schedule=config.example_file_nus_sampling_schedule,
        outfile=out_dir / "nus.ft1",
        max_points=192,
        f1180=True,
        shift=False,
    )

    print("\n5/7: Running methyl 1H-13C example.")
    run_methyl(
        infile=config.example_file_non_deuterated,
        outdir=out_dir,
        outfile="methyl.ft2",
        min_1H=-0.5,
        max_1H=2.5,
        p0=151.0,
        alt=True,
        neg=True,
    )
    print("\n6/7: Running aromatic side chain FID-Net2 example.")
    run_aromatic(
        infile = config.example_file_aromatic,
        outfile = out_dir / "aromatic.ft2",
        UseGPU = True,
        GPUIDX = None,
        offset1h = 0.4,
        offset13c = 0.4,
    )
    
    if skip_3d:
        print(
            "\n7/7: Skipping 3D HNCA example because it takes a"
            "long time. Use --no-skip-3d to run it."
        )
    else:
        print("\n7/7: Running methyl 1H-13C example.")
        run_3d_hnca(
            config.example_file_hnca,
            str(out_dir / "hnca.ft1"),
        )
