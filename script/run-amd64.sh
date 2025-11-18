#!/bin/bash

# Script to run AMD64 Docker container for Cochl Edge SDK
# Usage: ./script/run-amd64.sh [command]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
IMAGE_NAME="${IMAGE_NAME:-cochl-edge-sdk}"
IMAGE_TAG="${IMAGE_TAG:-amd64-latest}"
CONTAINER_NAME="${CONTAINER_NAME:-cochl-edge-sdk-amd64}"
PLATFORM="linux/amd64"

echo "=================================================="
echo "Cochl Edge SDK - AMD64 Docker Runner"
echo "=================================================="
echo "Image: $IMAGE_NAME:$IMAGE_TAG"
echo "Container: $CONTAINER_NAME"
echo ""

cd "$PROJECT_ROOT"

# Check if image exists
if ! docker image inspect "$IMAGE_NAME:$IMAGE_TAG" &>/dev/null; then
    echo "Error: Docker image $IMAGE_NAME:$IMAGE_TAG not found"
    echo ""
    echo "Please build the image first:"
    echo "  cd docker"
    echo "  docker build --platform linux/amd64 -f Dockerfile.linux-amd64 -t $IMAGE_NAME:$IMAGE_TAG .."
    exit 1
fi

# Remove existing container if running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Removing existing container: $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
fi

# Parse command
COMMAND="$@"

# Build docker run arguments
RUN_ARGS="--platform $PLATFORM"
RUN_ARGS="$RUN_ARGS --name $CONTAINER_NAME"
RUN_ARGS="$RUN_ARGS --rm"
RUN_ARGS="$RUN_ARGS -it"
RUN_ARGS="$RUN_ARGS -v $(pwd):/workspace"
RUN_ARGS="$RUN_ARGS -w /workspace"

# Set environment variables
RUN_ARGS="$RUN_ARGS -e LD_LIBRARY_PATH=/workspace/api/build:/opt/libtorch/lib:/opt/tvm/lib:/opt/tflite/tflite-dist/lib:/usr/lib/llvm-14/lib"
RUN_ARGS="$RUN_ARGS -e PYTHONPATH=/opt/tvm/python:/workspace/python"

# Run container
if [ -z "$COMMAND" ]; then
    # Interactive shell
    echo "Starting interactive shell..."
    echo ""
    docker run $RUN_ARGS "$IMAGE_NAME:$IMAGE_TAG" /bin/bash
else
    # Run specific command
    echo "Running: $COMMAND"
    echo ""
    docker run $RUN_ARGS "$IMAGE_NAME:$IMAGE_TAG" $COMMAND
fi
