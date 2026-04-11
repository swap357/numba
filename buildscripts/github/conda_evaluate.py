#!/usr/bin/env python

import json
import os
from pathlib import Path


CONDA_BUILD_MATRIX = {
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

CONDA_TEST_MATRIX = {
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
        {"python-version": "3.14", "numpy_test": "2.4"},
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
        {"python-version": "3.14", "numpy_test": "2.4"},
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
        {"python-version": "3.14", "numpy_test": "2.4"},
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
        {"python-version": "3.14", "numpy_test": "2.4"},
    ],
}


def add_platform(matrix, platform):
    """Add platform field to each matrix entry."""
    return [dict(item, platform=platform) for item in matrix]


def evaluate(event, label, platform, inputs="{}"):
    """Return (build_matrix, test_matrix) for the given parameters."""
    base_build = CONDA_BUILD_MATRIX.get(platform, [])
    base_test = CONDA_TEST_MATRIX.get(platform, [])

    if event in ("pull_request", "push"):
        build_matrix = base_build
        test_matrix = base_test
    elif event == "label" and label == "build_numba_conda":
        build_matrix = base_build
        test_matrix = base_test
    elif event == "workflow_dispatch":
        params = json.loads(inputs)
        build_matrix = list(base_build)
        test_matrix = list(base_test)

        python_version = params.get("python_version")
        if python_version:
            build_matrix = [
                item for item in build_matrix
                if item["python-version"] == python_version
            ]
            test_matrix = [
                item for item in test_matrix
                if item["python-version"] == python_version
            ]

        numpy_build_version = params.get("numpy_build_version")
        if numpy_build_version:
            build_matrix = [
                item for item in build_matrix
                if item["numpy_build"] == numpy_build_version
            ]

        numpy_test_version = params.get("numpy_test_version")
        if numpy_test_version:
            test_matrix = [
                item for item in test_matrix
                if item["numpy_test"] == numpy_test_version
            ]
    else:
        build_matrix = []
        test_matrix = []

    build_matrix = add_platform(build_matrix, platform)
    test_matrix = add_platform(test_matrix, platform)
    return build_matrix, test_matrix


if __name__ == "__main__":
    event = os.environ.get("GITHUB_EVENT_NAME")
    label = os.environ.get("GITHUB_LABEL_NAME")
    inputs = os.environ.get("GITHUB_WORKFLOW_INPUT", "{}")
    platform = os.environ.get("GITHUB_PLATFORM")

    print(
        "Deciding what to do based on event: "
        f"'{event}', label: '{label}', "
        f"inputs: '{inputs}', platform: '{platform}'"
    )

    build_matrix, test_matrix = evaluate(
        event, label, platform, inputs,
    )

    build_json = json.dumps(build_matrix)
    test_json = json.dumps(test_matrix)

    print(f"Build Matrix JSON: {build_json}")
    print(f"Test Matrix JSON: {test_json}")

    Path(os.environ["GITHUB_OUTPUT"]).write_text(
        f"build-matrix-json={build_json}\n"
        f"test-matrix-json={test_json}\n"
    )
