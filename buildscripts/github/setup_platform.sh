#!/bin/bash

set -x

PLATFORM=$1
BUILD_ENV=${2:-"build"}  # Default to "build" if not specified

echo "Setting up platform-specific requirements for ${PLATFORM} (environment: ${BUILD_ENV})"

case "${PLATFORM}" in
    "osx-64")
        echo "Setting up macOS SDK for osx-64 build"
        if [ "${BUILD_ENV}" == "build" ]; then
            sdk_dir="buildscripts/github"
            mkdir -p "${sdk_dir}"

            # Download SDK
            echo "Downloading MacOSX10.10.sdk.tar.xz"
            wget -q https://github.com/phracker/MacOSX-SDKs/releases/download/11.3/MacOSX10.10.sdk.tar.xz

            # Verify checksum
            echo "Verifying SDK checksum"
            shasum -a 256 -c "${sdk_dir}/MacOSX10.10.sdk.checksum"

            # Extract SDK to /opt
            echo "Extracting SDK to /opt"
            sudo mkdir -p /opt
            sudo tar -xf MacOSX10.10.sdk.tar.xz -C /opt
            echo "macOS SDK setup complete"
            
            # Install OpenMP via conda only for build env
            conda install --yes llvm-openmp
            # flag to help linker find libomp
            export LDFLAGS="-L$CONDA_PREFIX/lib -lomp"
        fi
        ;;
    
    "osx-arm64")
        echo "Setting up for osx-arm64 build"
        if [ "${BUILD_ENV}" == "build" ]; then
            # Install OpenMP via conda only for build env
            conda install --yes llvm-openmp
            # flag to help linker find libomp
            export LDFLAGS="-L$CONDA_PREFIX/lib -lomp"
        fi
        ;;
    
    "linux-64")
        echo "Setting up Linux dependencies"
        
        # Install TBB based on environment
        if [ "${BUILD_ENV}" == "build" ]; then
            echo "Installing TBB-devel for build environment"
            python -m pip install tbb-devel==2021.6
        elif [ "${BUILD_ENV}" == "test" ]; then
            echo "Installing TBB for test environment"
            python -m pip install tbb==2021.6
        fi
        ;;
    
    "linux-aarch64")
        ;;
    
    "win-64")
        echo "Setting up Windows dependencies"
        
        # Install TBB based on environment
        if [ "${BUILD_ENV}" == "build" ]; then
            echo "Installing TBB-devel for build environment"
            python -m pip install tbb==2021.6 tbb-devel==2021.6
        elif [ "${BUILD_ENV}" == "test" ]; then
            echo "Installing TBB for test environment"
            python -m pip install tbb==2021.6
        fi
        ;;
    
    *)
        echo "No specific setup required for platform: ${PLATFORM}"
        ;;
esac

echo "Platform setup complete for ${PLATFORM} (environment: ${BUILD_ENV})" 
