#!/usr/bin/env python

import json
import os
from pathlib import Path


event = os.environ.get("GITHUB_EVENT_NAME")
label = os.environ.get("GITHUB_LABEL_NAME")
inputs = os.environ.get("GITHUB_WORKFLOW_INPUT", "{}")
platform = os.environ.get("GITHUB_PLATFORM")

# Conda build matrix by platform
conda_build_matrix = {
    "linux-64": [
        {"python-version": "3.10", "numpy_build": "2.0"},
        {"python-version": "3.11", "numpy_build": "2.0"},
        {"python-version": "3.12", "numpy_build": "2.0"},
        {"python-version": "3.13", "numpy_build": "2.1"},
        {"python-version": "3.14", "numpy_build": "2.3"},
    ],
    "linux-aarch64": [
        {"python-version": "3.10", "numpy_build": "2.0"},
        {"python-version": "3.11", "numpy_build": "2.0"},
        {"python-version": "3.12", "numpy_build": "2.0"},
        {"python-version": "3.13", "numpy_build": "2.1"},
        {"python-version": "3.14", "numpy_build": "2.3"},
    ],
    "osx-arm64": [
        {"python-version": "3.10", "numpy_build": "2.0"},
        {"python-version": "3.11", "numpy_build": "2.0"},
        {"python-version": "3.12", "numpy_build": "2.0"},
        {"python-version": "3.13", "numpy_build": "2.1"},
        {"python-version": "3.14", "numpy_build": "2.3"},
    ],
    "win-64": [
        {"python-version": "3.10", "numpy_build": "2.0"},
        {"python-version": "3.11", "numpy_build": "2.0"},
        {"python-version": "3.12", "numpy_build": "2.0"},
        {"python-version": "3.13", "numpy_build": "2.1"},
        {"python-version": "3.14", "numpy_build": "2.3"},
    ],
}

# Conda test matrix by platform
conda_test_matrix = {
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
        {"python-version": "3.14", "numpy_test": "2.3"},
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
        {"python-version": "3.14", "numpy_test": "2.3"},
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
        {"python-version": "3.14", "numpy_test": "2.3"},
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
        {"python-version": "3.14", "numpy_test": "2.3"},
    ],
}


def add_platform(matrix, platform):
    """Add platform field to each matrix entry."""
    return [dict(item, platform=platform) for item in matrix]

print(
    "Deciding what to do based on event: "
    f"'{event}', label: '{label}', inputs: '{inputs}', platform: '{platform}'"
)

# Get platform-specific matrices
base_build_matrix = conda_build_matrix.get(platform, [])
base_test_matrix = conda_test_matrix.get(platform, [])

if event in ("pull_request", "push"):
    # This condition is entered on pull requests and pushes. The controlling
    # workflow is expected to filter push events to only the `main` branch.
    print(f"{event} detected, running full build matrix.")
    build_matrix = base_build_matrix
    test_matrix = base_test_matrix
elif event == "label" and label == "build_numba_conda":
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

# Add platform to both matrices
build_matrix = add_platform(build_matrix, platform)
test_matrix = add_platform(test_matrix, platform)

build_json = json.dumps(build_matrix)
test_json = json.dumps(test_matrix)

print(f"Build Matrix JSON: {build_json}")
print(f"Test Matrix JSON: {test_json}")

Path(os.environ["GITHUB_OUTPUT"]).write_text(
    f"build-matrix-json={build_json}\n"
    f"test-matrix-json={test_json}\n"
)

