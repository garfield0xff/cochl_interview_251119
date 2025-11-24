#!/bin/bash

# Check if TVM directory exists
if [ ! -d "tvm" ]; then
    echo "Error: TVM directory not found."
    echo "To install TVM, run the following commands:"
    echo "  git clone --recursive https://github.com/apache/tvm.git"
    exit 1
fi

# Check if TVM is built
if [ ! -f "tvm/build-arm64/libtvm.so" ]; then
    echo "Error: TVM is not built."
    echo "To build TVM, run the following:"
    echo "  cd tvm && mkdir build-arm64 && cd build-arm64"
    echo "  cmake .. && make -j4"
    exit 1
fi

echo "=== Installing TVM Python dependencies ==="

# Install tvm-ffi
if [ ! -d "tvm/3rdparty/tvm-ffi" ]; then
    echo "Error: tvm-ffi directory not found."
    exit 1
fi

cd tvm/3rdparty/tvm-ffi && pip install . && cd ../../..

# Install Python dependencies
pip install psutil decorator attrs cloudpickle xgboost ml_dtypes

# Set environment variables
export PYTHONPATH=/workspace/tvm/python:$PYTHONPATH
export LD_LIBRARY_PATH=/workspace/tvm/build-arm64:$LD_LIBRARY_PATH

echo ""
echo "=== Testing TVM installation ==="
python3 -c "import tvm; print(f'✓ TVM version: {tvm.__version__}')" && \
echo "✓ TVM environment successfully configured!"