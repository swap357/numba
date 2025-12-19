"""Centralized configuration for GitHub Actions workflows."""

# Environment variables
CONDA_CHANNEL_NUMBA = "numba/label/dev"
WHEELS_INDEX_URL = "https://pypi.anaconda.org/numba/label/dev/simple"
ARTIFACT_RETENTION_DAYS = 7

# Python version for conda environment setup (not the build matrix)
CONDA_SETUP_PYTHON_VERSION = "3.12"

# Extra channels mapping (python version -> channel)
EXTRA_CHANNELS_MAP = {
    "3.14": "-c ad-testing/label/py314",
}

# Platform-specific configuration
PLATFORMS = {
    "linux-64": {
        "runner": "ubuntu-latest",
        "manylinux_image": "quay.io/pypa/manylinux2014_x86_64",
        "use_tbb": True,
        "file_uri_prefix": "file://",
    },
    "linux-aarch64": {
        "runner": "ubuntu-24.04-arm",
        "manylinux_image": "quay.io/pypa/manylinux_2_28_aarch64",
        "use_tbb": False,
        "file_uri_prefix": "file://",
    },
    "osx-arm64": {
        "runner": "macos-14",
        "use_tbb": True,
        "file_uri_prefix": "file://",
    },
    "win-64": {
        "runner": "windows-2025",
        "use_tbb": True,
        "file_uri_prefix": "",  # Windows doesn't use file:// prefix
    },
}

# Dependencies
DEPENDENCIES = {
    "conda_build": "conda-build",
    "conda_libmamba_solver": '"conda-libmamba-solver<25.11"',
    "tbb_version": "2021.6",
}


def get_extra_channels(python_version: str) -> str:
    """Get extra channels for a given Python version."""
    return EXTRA_CHANNELS_MAP.get(python_version, "")


def get_platform_config(platform: str) -> dict:
    """Get platform-specific configuration."""
    return PLATFORMS.get(platform, {})

