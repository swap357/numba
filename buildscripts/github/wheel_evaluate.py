#!/usr/bin/env python

import json
import os
import sys
from pathlib import Path

# Add parent directory to path to import workflow_config
sys.path.insert(0, str(Path(__file__).parent))
from workflow_config import (
    CONDA_CHANNEL_NUMBA,
    WHEELS_INDEX_URL,
    ARTIFACT_RETENTION_DAYS,
    get_platform_config,
)


event = os.environ.get("GITHUB_EVENT_NAME")
label = os.environ.get("GITHUB_LABEL_NAME")
inputs = os.environ.get("GITHUB_WORKFLOW_INPUT", "{}")
platform = os.environ.get("GITHUB_PLATFORM")

# Wheel build matrix by platform
wheel_build_matrix = {
    "linux-64": [
        {"python-version": "3.10", "numpy_build": "2.0.2", "python_tag": "cp310"},
        {"python-version": "3.11", "numpy_build": "2.0.2", "python_tag": "cp311"},
        {"python-version": "3.12", "numpy_build": "2.0.2", "python_tag": "cp312"},
        {"python-version": "3.13", "numpy_build": "2.1.3", "python_tag": "cp313"},
        {"python-version": "3.14", "numpy_build": "2.3.3", "python_tag": "cp314"},
    ],
    "linux-aarch64": [
        {"python-version": "3.10", "numpy_build": "2.0.2", "python_tag": "cp310"},
        {"python-version": "3.11", "numpy_build": "2.0.2", "python_tag": "cp311"},
        {"python-version": "3.12", "numpy_build": "2.0.2", "python_tag": "cp312"},
        {"python-version": "3.13", "numpy_build": "2.1.3", "python_tag": "cp313"},
        {"python-version": "3.14", "numpy_build": "2.3.3", "python_tag": "cp314"},
    ],
    "osx-arm64": [
        {"python-version": "3.10", "numpy_build": "2.0.2", "python_tag": "cp310"},
        {"python-version": "3.11", "numpy_build": "2.0.2", "python_tag": "cp311"},
        {"python-version": "3.12", "numpy_build": "2.0.2", "python_tag": "cp312"},
        {"python-version": "3.13", "numpy_build": "2.1.3", "python_tag": "cp313"},
        {"python-version": "3.14", "numpy_build": "2.3.3", "python_tag": "cp314"},
    ],
    "win-64": [
        {"python-version": "3.10", "numpy_build": "2.0.2", "python_tag": "cp310"},
        {"python-version": "3.11", "numpy_build": "2.0.2", "python_tag": "cp311"},
        {"python-version": "3.12", "numpy_build": "2.0.2", "python_tag": "cp312"},
        {"python-version": "3.13", "numpy_build": "2.1.3", "python_tag": "cp313"},
        {"python-version": "3.14", "numpy_build": "2.3.3", "python_tag": "cp314"},
    ],
}

# Wheel test matrix by platform
wheel_test_matrix = {
    "linux-64": [
        {"python-version": "3.10", "numpy_test": "1.23"},
        {"python-version": "3.10", "numpy_test": "1.24"},
        {"python-version": "3.10", "numpy_test": "1.25"},
        {"python-version": "3.11", "numpy_test": "1.26"},
        {"python-version": "3.11", "numpy_test": "2.0"},
        {"python-version": "3.11", "numpy_test": "2.2"},
        {"python-version": "3.12", "numpy_test": "1.26"},
        {"python-version": "3.12", "numpy_test": "2.0"},
        {"python-version": "3.12", "numpy_test": "2.2"},
        {"python-version": "3.13", "numpy_test": "2.2"},
        {"python-version": "3.13", "numpy_test": "2.3"},
        {"python-version": "3.14", "numpy_test": "2.3.3"},
    ],
    "linux-aarch64": [
        {"python-version": "3.10", "numpy_test": "1.23"},
        {"python-version": "3.10", "numpy_test": "1.24"},
        {"python-version": "3.10", "numpy_test": "1.25"},
        {"python-version": "3.11", "numpy_test": "1.26"},
        {"python-version": "3.11", "numpy_test": "2.0"},
        {"python-version": "3.11", "numpy_test": "2.2"},
        {"python-version": "3.12", "numpy_test": "1.26"},
        {"python-version": "3.12", "numpy_test": "2.0"},
        {"python-version": "3.12", "numpy_test": "2.2"},
        {"python-version": "3.13", "numpy_test": "2.2"},
        {"python-version": "3.13", "numpy_test": "2.3"},
        {"python-version": "3.14", "numpy_test": "2.3.3"},
    ],
    "osx-arm64": [
        {"python-version": "3.10", "numpy_test": "1.23"},
        {"python-version": "3.10", "numpy_test": "1.24"},
        {"python-version": "3.10", "numpy_test": "1.25"},
        {"python-version": "3.11", "numpy_test": "1.26"},
        {"python-version": "3.11", "numpy_test": "2.0"},
        {"python-version": "3.11", "numpy_test": "2.2"},
        {"python-version": "3.12", "numpy_test": "1.26"},
        {"python-version": "3.12", "numpy_test": "2.0"},
        {"python-version": "3.12", "numpy_test": "2.2"},
        {"python-version": "3.13", "numpy_test": "2.2"},
        {"python-version": "3.13", "numpy_test": "2.3"},
        {"python-version": "3.14", "numpy_test": "2.3.3"},
    ],
    "win-64": [
        {"python-version": "3.10", "numpy_test": "1.23"},
        {"python-version": "3.10", "numpy_test": "1.24"},
        {"python-version": "3.10", "numpy_test": "1.25"},
        {"python-version": "3.11", "numpy_test": "1.26"},
        {"python-version": "3.11", "numpy_test": "2.0"},
        {"python-version": "3.11", "numpy_test": "2.2"},
        {"python-version": "3.12", "numpy_test": "1.26"},
        {"python-version": "3.12", "numpy_test": "2.0"},
        {"python-version": "3.12", "numpy_test": "2.2"},
        {"python-version": "3.13", "numpy_test": "2.2"},
        {"python-version": "3.13", "numpy_test": "2.3"},
        {"python-version": "3.14", "numpy_test": "2.3.3"},
    ],
}


def add_canonical_version(matrix):
    """Add python_canonical field (e.g., '3.10' -> '3.10')"""
    result = []
    for item in matrix:
        item_copy = item.copy()
        py_version = item["python-version"]
        # Extract major.minor (e.g., "3.10" from "3.10" or "3.14")
        parts = py_version.split(".")
        item_copy["python_canonical"] = ".".join(parts[:2])
        result.append(item_copy)
    return result


def add_platform(matrix, platform):
    """Add platform field to each matrix entry."""
    return [dict(item, platform=platform) for item in matrix]


print(
    "Deciding what to do based on event: "
    f"'{event}', label: '{label}', inputs: '{inputs}', platform: '{platform}'"
)

# Get platform-specific matrices
base_build_matrix = wheel_build_matrix.get(platform, [])
base_test_matrix = wheel_test_matrix.get(platform, [])

if event in ("pull_request", "push"):
    # This condition is entered on pull requests and pushes. The controlling
    # workflow is expected to filter push events to only the `main` branch.
    print(f"{event} detected, running full build matrix.")
    build_matrix = base_build_matrix
    test_matrix = base_test_matrix
elif event == "label" and label == "build_numba_wheel":
    print("build label detected")
    build_matrix = base_build_matrix
    test_matrix = base_test_matrix
elif event == "workflow_dispatch":
    print("workflow_dispatch detected")
    params = json.loads(inputs)
    build_matrix = base_build_matrix
    test_matrix = base_test_matrix

    # Filter by python_version if specified
    python_version = params.get("python_version")
    if python_version:
        print(f"Filtering by python_version: {python_version}")
        build_matrix = [
            item for item in build_matrix
            if item["python-version"] == python_version
        ]
        test_matrix = [
            item for item in test_matrix
            if item["python-version"] == python_version
        ]

    # Filter by numpy_version if specified (applies to both build and test)
    numpy_build_version = params.get("numpy_build_version")
    if numpy_build_version:
        print(f"Filtering build matrix by numpy_build_version: {numpy_build_version}")
        build_matrix = [
            item for item in build_matrix
            if item["numpy_build"] == numpy_build_version
        ]

    numpy_test_version = params.get("numpy_test_version")
    if numpy_test_version:
        print(f"Filtering test matrix by numpy_test_version: {numpy_test_version}")
        test_matrix = [
            item for item in test_matrix
            if item["numpy_test"] == numpy_test_version
        ]
else:
    # For any other events, produce an empty matrix.
    build_matrix = []
    test_matrix = []

# Add canonical python versions and platform to both matrices
build_matrix_extended = add_canonical_version(build_matrix)
test_matrix_extended = add_canonical_version(test_matrix)
build_matrix_extended = add_platform(build_matrix_extended, platform)
test_matrix_extended = add_platform(test_matrix_extended, platform)

# Get platform config
platform_config = get_platform_config(platform)
config = {
    "runner": platform_config.get("runner", "ubuntu-latest"),
    "conda_channel_numba": CONDA_CHANNEL_NUMBA,
    "wheels_index_url": WHEELS_INDEX_URL,
    "artifact_retention_days": ARTIFACT_RETENTION_DAYS,
    "manylinux_image": platform_config.get("manylinux_image", ""),
    "use_tbb": platform_config.get("use_tbb", True),
}

build_json = json.dumps(build_matrix_extended)
test_json = json.dumps(test_matrix_extended)
config_json = json.dumps(config)

print(f"Build Matrix JSON: {build_json}")
print(f"Test Matrix JSON: {test_json}")
print(f"Config JSON: {config_json}")

Path(os.environ["GITHUB_OUTPUT"]).write_text(
    f"build-matrix-json={build_json}\n"
    f"test-matrix-json={test_json}\n"
    f"config-json={config_json}\n"
)

