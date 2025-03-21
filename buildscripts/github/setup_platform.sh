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
        else
            # For test environment
            if command -v brew &> /dev/null; then
                brew install libomp
            fi
        fi
        ;;
    
    "osx-arm64")
        echo "Setting up for osx-arm64 build"
        if [ "${BUILD_ENV}" == "build" ]; then
            # Install OpenMP via conda only for build env
            conda install --yes llvm-openmp
        else
            # For test environment
            if command -v brew &> /dev/null; then
                brew install libomp
            fi
        fi
        ;;
    
    "linux-64")
        echo "Setting up Linux dependencies"
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y build-essential libomp-dev
        fi
        
        # Install TBB for Linux-64
        echo "Installing TBB for Linux-64"
        python -m pip install tbb==2021.6 tbb-devel==2021.6
        ;;
    
    "linux-aarch64")
        
        ;;
    
    "win-64")
        echo "Setting up Windows dependencies"
        
        # Install TBB for Windows
        echo "Installing TBB for Windows"
        python -m pip install tbb==2021.6 tbb-devel==2021.6
        ;;
    
    *)
        echo "No specific setup required for platform: ${PLATFORM}"
        ;;
esac

echo "Platform setup complete for ${PLATFORM} (environment: ${BUILD_ENV})" 
