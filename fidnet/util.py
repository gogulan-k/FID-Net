import hashlib
import tarfile
import urllib.request
from pathlib import Path

from fidnet import config


def download_example_data(force: bool = False):
    example_files = [
        config.example_file_non_deuterated,
        config.example_file_hnca,
        config.example_file_nus_reconstruct,
        config.example_file_nus_sampling_schedule,
        config.example_file_ctcp,
        config.example_file_con_decouple,
        config.example_file_ca_detect,
        config.example_file_aromatic,
    ]
    # ALl important input files are present
    if all([example_file.exists() for example_file in example_files]) and not force:
        return
    # Make sure the dir exists
    config.example_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading example data from {config.example_url}")
    urllib.request.urlretrieve(  # nosec
        config.example_url, config.example_dir.parent / "example.tar.gz"
    )  # nosec
    # Extract the tar file
    print(f"Extracting example data to {config.example_dir}")
    with tarfile.open(config.example_dir.parent / "example.tar.gz") as tar:
        tar.extractall(config.example_dir.parent)


def download_all_weights(force: bool = False):
    for path in config.weight_checksums:
        download_weights(path, force=force)


def download_weights(path: Path, force: bool = False):
    # Only download when necessary
    if not force and path.exists() and check_weight_checksum(path):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    url = config.weights_url + str(path.name)
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, path)  # nosec


def check_weight_checksum(path: Path):
    """
    Check if the checksum of the data is equal to the given checksum.
    """
    checksum = config.weight_checksums[path]
    return check_checksum(path, checksum)


def check_checksum(path: Path, checksum: str):
    with path.open(mode="rb") as f:
        data = f.read()
    return hashlib.md5(data).hexdigest() == checksum  # nosec
