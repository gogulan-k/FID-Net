"""FID Net Wrapper."""

import warnings

try:
    from fidnet._version import __version__
except ImportError:
    __version__ = "not-installed"
    warnings.warn(
        "You are running a non-installed version of fidnet."
        "If you are running this from a git repo, please run"
        "`pip install -e .` to install the package."
    )


from fidnet._config import Settings as _Settings

config = _Settings()
