"""
This file can be used to configure the project. The different
attributes of the "Settings" class can be accessed as follows:
from  fidnet import config.

A typical use case for this file is to keep track of paths
to different files in the project.

This is using pydantic-settings. In order to read more about how
to use this, see: https://docs.pydantic.dev/latest/concepts/pydantic_settings/

One way of overriding the default settings is to create a .env file anywhere
in the file tree above the path you are running the code from. For example,
in the root of the project or in your home directory.

The .env file should contain the variables you want to override with the
"fidnet_" prefix, for example:

fidnet_DATA_DIR=/path/to/data

Because you can have different .env files in different machines, it
makes it easy to have different settings (think paths) for
different machines.

You don't have to use this mechanism, but often it makes life easier.

"""

from pathlib import Path

from dotenv import find_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

_here = Path(__file__).parent
_repo_root = _here.parent


class RootSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="fidnet_",
        env_file=find_dotenv(".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ROOT_DIR: Path = _repo_root

    def __str__(self) -> str:
        return super().__str__().replace(" ", "\n")


class DataDirSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="fidnet_",
        env_file=find_dotenv(".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    _root_settings: RootSettings = RootSettings()

    DATA_DIR: Path = _root_settings.ROOT_DIR / "data"

    def __str__(self) -> str:
        return super().__str__().replace(" ", "\n")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="fidnet_",
        env_file=find_dotenv(".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    _data_dir_settings: DataDirSettings = DataDirSettings()

    _data_dir: Path = _data_dir_settings.DATA_DIR
    _data_dir.mkdir(parents=True, exist_ok=True)

    # Add subpaths to different files and directories here
    # in one central place instead of spreading them out

    base_url: str = "https://github.com/gogulan-k/FID-Net/releases/download/v0.51-alpha/"
    weights_url: str = base_url
    example_url: str = "https://github.com/gogulan-k/FID-Net/releases/download/v0.51-alpha/example.tar.gz"
    # Weights
    weights_dir: Path = _data_dir / "weights"
    weights_1h_methyl: Path = weights_dir / "fidnet_1h_methyl.h5"
    weights_13c_methyl: Path = weights_dir / "fidnet_13c_methyl.h5"
    weights_3D_HNCA_decouple: Path = weights_dir / "fidnet_3dca_decouple.h5"
    weights_nus_reconstruct: Path = weights_dir / "fidnet_recon.h5"
    weights_ctcp: Path = weights_dir / "jcoup_2dctcp.h5"
    weights_con_decouple: Path = weights_dir / "jcoup_2dcon.h5"
    weights_ca_detect: Path = weights_dir / "jcoup_2dcadet.h5"
    weights_aromatic: Path = weights_dir / "Aromatic_weights.h5"

    weight_checksums: dict[Path, str] = {
        weights_1h_methyl: "a95b14795bbd9ef05604d6677f99362e",
        weights_13c_methyl: "8dd99bed02a90561878c147d4819b3b5",
        weights_3D_HNCA_decouple: "81827494c797bfc8062d3f5790863c17",
        weights_nus_reconstruct: "c29c79e1e5a012a8c27a995025743f28",
        weights_ctcp: "7f181904769939dc1f05e5ef58a3f1d6",
        weights_con_decouple: "ba1e3699a7cd7d48a682912c1497c7be",
        weights_ca_detect: "40f28f58cd0a714c88a56a3426b7dd7d",
        weights_aromatic:"e403c9f2f97e80d0a9ce82c6483215a1",
    }

    example_dir: Path = _data_dir / "example"

    example_file_non_deuterated: Path = example_dir / "methyl" / "test.fid"
    example_file_hnca: Path = example_dir / "hnca_decouple" / "test.ft2"
    example_file_nus_reconstruct: Path = example_dir / "nus" / "hdac_ex.ft1"
    example_file_nus_sampling_schedule: Path = example_dir / "nus" / "ss_hdac.out"
    example_file_ctcp: Path = example_dir / "ctcp_decouple" / "test001.ft1"
    example_file_con_decouple: Path = example_dir / "con_decouple" / "test001.ft1"
    example_file_ca_detect: Path = example_dir / "ca_detect" / "test.ft1"
    example_file_aromatic: Path = example_dir / "aromatic" / "test.ft1"

    def __str__(self):
        return (
            str(self._data_dir_settings) + "\n" + super().__str__().replace(" ", "\n")
        )
