#!/bin/bash
set -ex

# Repairs and patches a Numba wheel inside the manylinux container
# Usage: repair_wheel_linux-64.sh <python_tag> <use_tbb>

PYTHON_TAG=$1
USE_TBB=${2:-"true"}
WHEEL_DIR=${3:-"/io/wheelhouse"}

# Debug output
echo "PYTHON_TAG: $PYTHON_TAG"
echo "USE_TBB: $USE_TBB"
echo "WHEEL_DIR: $WHEEL_DIR"

# Set Python path
PYTHON_PATH=/opt/python/${PYTHON_TAG}-${PYTHON_TAG}/bin/python

# Install required tools
$PYTHON_PATH -m pip install auditwheel patchelf twine wheel

# Make sure the wheelhouse directory exists
mkdir -p $WHEEL_DIR

# Find the wheel
cd $WHEEL_DIR
echo "Contents of wheel directory ($WHEEL_DIR):"
ls -la

WHEEL_FILE=$(ls -1 numba*.whl 2>/dev/null | head -1)

if [ -z "$WHEEL_FILE" ]; then
    echo "No wheel file found in $WHEEL_DIR"
    exit 1
fi

echo "Repairing wheel: $WHEEL_FILE"

# Create a temporary directory for repair
REPAIR_DIR=$(mktemp -d)
cp "$WHEEL_FILE" "$REPAIR_DIR/"
cd "$REPAIR_DIR"

# Repair with auditwheel
$PYTHON_PATH -m auditwheel repair "$WHEEL_FILE"
cd wheelhouse

# Get the filename of the wheel to be patched
WHEEL_PATCHED=$(ls -1 *.whl | head -1)

# Unpack the wheel for patching
$PYTHON_PATH -m wheel unpack $WHEEL_PATCHED
WHEEL_DIR_UNPACKED=$(find . -maxdepth 1 -type d -name "numba-*" | head -1)
cd "$WHEEL_DIR_UNPACKED"

# Patch libraries
if [ -d "numba.libs" ]; then
  cd numba.libs
  LIBTBB=$(ls libtbb* 2>/dev/null || echo "")
  LIBOMP=$(ls libgomp* 2>/dev/null || echo "")
  cd ..
  rm -rf numba.libs

  # Patch TBB libraries if present and TBB is enabled
  if [ "$USE_TBB" = "true" ] && [ -n "$LIBTBB" ]; then
    echo "Patching TBB libraries"
    TBBEXT=$(echo "$LIBTBB" | grep -oP "(\\.so.*)" || echo ".so")
    patchelf numba/np/ufunc/tbbpool*.so --replace-needed $LIBTBB libtbb$TBBEXT
    patchelf numba/np/ufunc/tbbpool*.so --remove-rpath
    ldd numba/np/ufunc/tbbpool*.so
    readelf -d numba/np/ufunc/tbbpool*.so
  fi

  # Patch OpenMP libraries if present
  if [ -n "$LIBOMP" ]; then
    echo "Patching OpenMP libraries"
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

# Clean the original wheel directory (remove the original wheel)
rm -f $WHEEL_DIR/numba*-linux_x86_64.whl

# Move repaired wheel back to output directory
cp *.whl $WHEEL_DIR/

# Verify wheel
$PYTHON_PATH -m twine check *.whl

echo "Wheel repair and patch completed successfully" 