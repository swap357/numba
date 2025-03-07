#!/bin/bash

if [[ "$(uname -s)" == *"Linux"* ]] && [[ "$(uname -p)" == *"86"* ]]; then
    EXTRA_BUILD_EXT_FLAGS="--werror --wall"
else
    EXTRA_BUILD_EXT_FLAGS=""
fi

if [[ "$(uname -s)" == *"Linux"* ]] && [[ "$(uname -p)" == *"ppc64le"* ]]; then
    # To workaround https://github.com/numba/numba/issues/7302 
    # because of a python build problem that the -pthread could be stripped.
    export CC="$CC -pthread"
    export CXX="$CXX -pthread"
fi

if [[ "$(uname -s)" == *"Darwin"* ]]; then
    # The following is suggested in https://docs.conda.io/projects/conda-build/en/latest/resources/compiler-tools.html?highlight=SDK#macos-sdk
    wget -q https://github.com/phracker/MacOSX-SDKs/releases/download/11.3/MacOSX10.10.sdk.tar.xz
    shasum -c ./buildscripts/incremental/MacOSX10.10.sdk.checksum
    tar -xf ./MacOSX10.10.sdk.tar.xz
    export SDKROOT=`pwd`/MacOSX10.10.sdk
    export CONDA_BUILD_SYSROOT=`pwd`/MacOSX10.10.sdk
    export macos_min_version=10.10
fi

MACOSX_DEPLOYMENT_TARGET=10.10 $PYTHON setup.py build_ext $EXTRA_BUILD_EXT_FLAGS build install --single-version-externally-managed --record=record.txt
