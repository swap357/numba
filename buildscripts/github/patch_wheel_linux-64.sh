#!/bin/bash
set -ex

# This script patches a Numba wheel to fix library dependencies
# Usage: patch_wheel_linux-64.sh <wheel_path> <python_path> <use_tbb>

WHEEL_PATH=$1
PYTHON_PATH=$2
USE_TBB=${3:-"true"}

WHEEL_DIR=$(dirname "$WHEEL_PATH")
cd "$WHEEL_DIR"

# Unpack the wheel
WHEEL_FILENAME=$(basename "$WHEEL_PATH")
$PYTHON_PATH -m wheel unpack "$WHEEL_FILENAME"
WHEEL_DIR_UNPACKED=$(find . -maxdepth 1 -type d -name "numba-*" | head -n 1)
cd "$WHEEL_DIR_UNPACKED"

if [ -d "numba.libs" ]; then
  cd numba.libs
  LIBTBB=$(ls libtbb* 2>/dev/null || echo "")
  LIBOMP=$(ls libgomp* 2>/dev/null || echo "")
  cd ..
  rm -rf numba.libs

  # Patch TBB libraries if present and TBB is enabled
  if [ "$USE_TBB" = "true" ] && [ -n "$LIBTBB" ]; then
    TBBEXT=$(echo "$LIBTBB" | grep -oP "(\\.so.*)" || echo ".so")
    patchelf numba/np/ufunc/tbbpool*.so --replace-needed $LIBTBB libtbb$TBBEXT
    patchelf numba/np/ufunc/tbbpool*.so --remove-rpath
    ldd numba/np/ufunc/tbbpool*.so
    readelf -d numba/np/ufunc/tbbpool*.so
  fi

  # Patch OpenMP libraries if present
  if [ -n "$LIBOMP" ]; then
    OMPEXT=$(echo "$LIBOMP" | grep -oP "(\\.so.*)" || echo ".so")
    patchelf numba/np/ufunc/omppool*.so --replace-needed $LIBOMP libgomp$OMPEXT
    patchelf numba/np/ufunc/omppool*.so --remove-rpath
    ldd numba/np/ufunc/omppool*.so
    readelf -d numba/np/ufunc/omppool*.so
  fi

  # Fix executable bit on scripts
  if [ -d "numba-*.data/scripts" ]; then
    chmod +x numba-*.data/scripts/*
  fi
fi

cd ..

# Repack the wheel
rm -f *.whl
$PYTHON_PATH -m wheel pack "$WHEEL_DIR_UNPACKED"

# Add -tbb suffix if TBB is enabled
if [ "$USE_TBB" = "true" ]; then
  WHEEL_NAME=$(ls numba-*.whl)
  NEW_WHEEL_NAME=${WHEEL_NAME/.whl/-tbb.whl}
  mv "$WHEEL_NAME" "$NEW_WHEEL_NAME"
fi

# Move the wheel back to the original location
FINAL_WHEEL=$(ls numba-*.whl)
mv "$FINAL_WHEEL" "$WHEEL_DIR/../"

echo "Wheel patching completed: $FINAL_WHEEL" 