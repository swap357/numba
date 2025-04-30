#!/bin/bash
set -ex

# Builds a Numba wheel inside the manylinux container
# Usage: build_wheel_linux-64.sh <python_tag> <numpy_version> <use_tbb>

PYTHON_TAG=$1
NUMPY_VERSION=$2
USE_TBB=${3:-"true"}
LLVMLITE_WHEEL_PATH=${4:-""}
WHEELS_INDEX_URL=${5:-"https://pypi.anaconda.org/numba/label/dev/simple"}

# Set Python path
PYTHON_PATH=/opt/python/${PYTHON_TAG}-${PYTHON_TAG}/bin/python

# Install dependencies
$PYTHON_PATH -m pip install build numpy==${NUMPY_VERSION} setuptools wheel

# Install TBB if enabled
if [ "$USE_TBB" = "true" ]; then
    $PYTHON_PATH -m pip install tbb==2021.6 tbb-devel==2021.6
fi

# Install llvmlite
if [ -n "$LLVMLITE_WHEEL_PATH" ] && [ -d "$LLVMLITE_WHEEL_PATH" ]; then
    $PYTHON_PATH -m pip install $LLVMLITE_WHEEL_PATH/*.whl
else
    $PYTHON_PATH -m pip install -i $WHEELS_INDEX_URL llvmlite
fi

# Change to the mounted workspace directory
cd /io

# Show directory contents for debugging
echo "Contents of /io directory:"
ls -la

# Build wheel from the workspace directory
$PYTHON_PATH -m build --wheel

# Build sdist based on python version (3.10)
if [ "$PYTHON_TAG" = "cp310" ]; then
    $PYTHON_PATH -m build --sdist
fi

# Create output directory if it doesn't exist
mkdir -p /io/wheelhouse

# Copy wheel to the output directory
cp dist/*.whl /io/wheelhouse/

echo "Wheel build completed successfully" 