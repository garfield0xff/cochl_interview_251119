# Cochl Interview
## Project Structure

```
.
├── api/                   # Core inference API (C/C++)
│   ├── include/
│   │   ├── cochl_api_c.h  # C API header
│   │   ├── runtime/       # Runtime manager
│   │   └── utils/         # Image utilities (stb_image, 
│   ├── src/
│   │   ├── cochl_api_c.cpp
│   │   └── runtime/       # TFLite/LibTorch implementations
│   └── test/              # Unit tests
├── sdk/                    # SDK wrapping the API
│   ├── include/
│   │   └── inference_engine.h
│   ├── src/
│   │   └── inference_engine.cpp
│   └── third_party/
│       └── cochl_api/     # Compiled API library
├── main/                   # Test application
│   ├── main.cpp
│   └── CMakeLists.txt
├── models/                 # Model files
│   ├── resnet50.tflite
│   └── resnet50.pt
└── third_party/           # Dependencies
    ├── tflite-arm64/
    └── libtorch-arm64/
```

## Build & Run Pipeline (ARM64)

### 1. Enter Docker Environment

```bash
cd docker
docker build --platform linux/arm64 -f Dockerfile.linux-arm64 -t chocl-edge-sdk::arm64-latest ..
cd ..
./script/run-arm64.sh
```

### 2. Build API (Shared Library)

```bash
cd /workspace/api/build
cmake -DBUILD_SHARED_LIBS=ON -DUSE_TFLITE=ON -DUSE_LIBTORCH=ON -DUSE_CUSTOM=ON ..
make cochl_api -j4

# Copy to SDK
cp lib/libcochl_api.so ../sdk/third_party/cochl_api/lib/
```

### 3. Build SDK

```bash
cd /workspace/sdk/build
cmake ..
make -j4
```

### 4. Build Main Application

```bash
cd /workspace/main/build
cmake ..
make -j4
```

### 5. Run Inference

```bash
# Set library path
export LD_LIBRARY_PATH=/workspace/third_party/libtorch-arm64/lib:/workspace/sdk/third_party/cochl_api/lib

# Run with TensorFlow Lite
./bin/main \
    /workspace/models/resnet50.tflite \
    /workspace/api/test/dog.png \
    /workspace/api/test/imagenet_class_index.json

# Run with LibTorch
./bin/main \
    /workspace/models/resnet50.pt \
    /workspace/api/test/dog.png \
    /workspace/api/test/imagenet_class_index.json

# Run with Custom Backend (Thread Pool - Mock)
# Note: Create a dummy model.bin file first if it doesn't exist
touch /workspace/models/model.bin
./bin/main \
    /workspace/models/model.bin \
    /workspace/api/test/dog.png \
    /workspace/api/test/imagenet_class_index.json
```

