#!/usr/bin/env python

import json
import os
from pathlib import Path


event = os.environ.get("GITHUB_EVENT_NAME")
label = os.environ.get("GITHUB_LABEL_NAME")
inputs_json = os.environ.get("GITHUB_WORKFLOW_INPUT", "{}")
inputs = json.loads(inputs_json)

runner_mapping = {
    "linux-64": "ubuntu-24.04",
    "linux-aarch64": "ubuntu-24.04-arm",
    "osx-64": "macos-13",
    "osx-arm64": "macos-14",
    "win-64": "windows-2019",
}

python_versions = ["3.10", "3.11", "3.12", "3.13"]
numpy_versions = {
    "3.10": "2.0.2",
    "3.11": "2.0.2",
    "3.12": "2.0.2",
    "3.13": "2.1.3"
}

# Create default matrices for all platforms
default_build_matrix = {"include": []}
default_test_matrix = {"include": []}

# Default platforms to build
default_platforms = ["linux-64", "osx-64", "osx-arm64", "win-64"]

# Generate matrices for all default platforms and Python versions
for platform in default_platforms:
    for py_ver in python_versions:
        np_ver = numpy_versions[py_ver]
        default_build_matrix["include"].append({
            "runner": runner_mapping[platform],
            "platform": platform,
            "python-version": py_ver,
            "numpy_build": np_ver
        })
        
        default_test_matrix["include"].append({
            "runner": runner_mapping[platform],
            "platform": platform,
            "python-version": py_ver,
            "numpy_test": np_ver
        })

print(
    "Deciding what to do based on event: "
    f"'{event}', label: '{label}', inputs: '{inputs_json}'"
)

build_matrix = {}
test_matrix = {}

if event == "pull_request":
    print("pull_request detected")
    build_matrix = default_build_matrix
    test_matrix = default_test_matrix
elif event == "label" and label == "build_wheels":
    print("build_wheels label detected")
    build_matrix = default_build_matrix
    test_matrix = default_test_matrix
elif event == "workflow_dispatch":
    print("workflow_dispatch detected")
    platform = inputs.get("platform", "all")
    
    if platform == "all":
        build_matrix = default_build_matrix
        test_matrix = default_test_matrix
    else:
        # Generate matrices for specified platform only
        build_matrix = {"include": []}
        test_matrix = {"include": []}
        
        for py_ver in python_versions:
            np_ver = numpy_versions[py_ver]
            build_matrix["include"].append({
                "runner": runner_mapping[platform],
                "platform": platform,
                "python-version": py_ver,
                "numpy_build": np_ver
            })
            
            test_matrix["include"].append({
                "runner": runner_mapping[platform],
                "platform": platform,
                "python-version": py_ver,
                "numpy_test": np_ver
            })
else:
    build_matrix = {"include": []}
    test_matrix = {"include": []}

print(f"Emitting build matrix:\n {json.dumps(build_matrix, indent=4)}")
print(f"Emitting test matrix:\n {json.dumps(test_matrix, indent=4)}")

output_path = Path(os.environ["GITHUB_OUTPUT"])
with output_path.open("w") as f:
    f.write(f"build_matrix={json.dumps(build_matrix)}\n")
    f.write(f"test_matrix={json.dumps(test_matrix)}") 
