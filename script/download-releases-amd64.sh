#!/bin/bash

# Script to download AMD64 (x64) build artifacts from GitHub Releases
# Usage: ./script/download-releases-amd64.sh [version]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default versions
TVM_VERSION="${TVM_VERSION:-0.22.0}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.1.0}"
TENSORFLOW_VERSION="${TENSORFLOW_VERSION:-2.15.0}"

# GitHub repository
GITHUB_REPO="${GITHUB_REPO:-garfield0xff/cochl_interview_251119}"

echo "=================================================="
echo "Downloading AMD64 Build Artifacts from GitHub"
echo "=================================================="
echo "Project Root: $PROJECT_ROOT"
echo "TVM Version: $TVM_VERSION"
echo "PyTorch Version: $PYTORCH_VERSION"
echo "TensorFlow Version: $TENSORFLOW_VERSION"
echo ""

cd "$PROJECT_ROOT"

# Create third_party directory
mkdir -p third_party

# Function to download and extract release asset
download_and_extract() {
    local tag=$1
    local filename=$2
    local extract_dir=$3

    echo "Downloading $filename from release $tag..."

    # Direct download from GitHub Releases
    local url="https://github.com/${GITHUB_REPO}/releases/download/${tag}/${filename}"

    if curl -L -f -o "/tmp/${filename}" "$url" 2>/dev/null; then
        echo "✓ Downloaded: $filename"

        # Extract to third_party directory
        if [ -n "$extract_dir" ]; then
            echo "  Extracting to $extract_dir..."
            mkdir -p "$extract_dir"
            tar -xzf "/tmp/${filename}" -C "$extract_dir"
            rm "/tmp/${filename}"
            echo "  ✓ Extracted successfully"
        fi
        return 0
    else
        echo "✗ Failed to download: $filename (not found or requires authentication)"
        return 1
    fi
}

# Download TVM x64 artifacts
echo "[1/6] Downloading TVM x64 C++ libraries..."
download_and_extract "tvm-v${TVM_VERSION}" "tvm-linux-x64-cpp.tar.gz" "third_party/tvm-amd64" || true

echo "[2/6] Downloading TVM x64 wheel..."
download_and_extract "tvm-v${TVM_VERSION}" "tvm-linux-x64-wheel.tar.gz" "third_party/tvm-amd64-wheel" || true

# Download LibTorch x64 artifacts
echo "[3/6] Downloading LibTorch x64 libraries..."
download_and_extract "libtorch-v${PYTORCH_VERSION}" "libtorch-linux-x64.tar.gz" "third_party" || true
# Rename libtorch to libtorch-amd64
if [ -d "third_party/libtorch" ]; then
    mv third_party/libtorch third_party/libtorch-amd64
    echo "  ✓ Renamed to libtorch-amd64"
fi

# Download TFLite x64 artifacts
echo "[4/6] Downloading TFLite x64 libraries..."
download_and_extract "tflite-v${TENSORFLOW_VERSION}" "tflite-linux-x64.tar.gz" "third_party" || true
# Rename tflite-dist to tflite-amd64
if [ -d "third_party/tflite-dist" ]; then
    mv third_party/tflite-dist third_party/tflite-amd64
    echo "  ✓ Renamed to tflite-amd64"
fi

echo ""
echo "=================================================="
echo "Download Summary"
echo "=================================================="

# Check which directories were created
for dir in third_party/tvm-amd64 third_party/libtorch-amd64 third_party/tflite-amd64; do
    if [ -d "$dir" ]; then
        size=$(du -sh "$dir" | cut -f1)
        echo "✓ $dir ($size)"
    else
        echo "✗ $dir (missing)"
    fi
done

echo ""
echo "=================================================="
echo "Libraries are ready for use!"
echo ""
echo "LibTorch AMD64:"
echo "  export LIBTORCH_HOME=$PROJECT_ROOT/third_party/libtorch-amd64"
echo "  export LD_LIBRARY_PATH=\$LIBTORCH_HOME/lib:\$LD_LIBRARY_PATH"
echo ""
echo "TFLite AMD64:"
echo "  export TFLITE_HOME=$PROJECT_ROOT/third_party/tflite-amd64"
echo "  export LD_LIBRARY_PATH=\$TFLITE_HOME/lib:\$LD_LIBRARY_PATH"
echo ""
echo "TVM AMD64:"
echo "  export TVM_HOME=$PROJECT_ROOT/third_party/tvm-amd64"
echo "  export LD_LIBRARY_PATH=\$TVM_HOME/lib:\$LD_LIBRARY_PATH"
