# Cochl Edge SDK

Edge device SDK for inference using Cochl API shared library.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                    SDK                          │
│  ┌──────────────────────────────────────────┐  │
│  │        Inference Engine                  │  │
│  │  ┌──────────┐        ┌───────────────┐  │  │
│  │  │ Profiler │        │  thread_pool  │  │  │
│  │  └──────────┘        └───────────────┘  │  │
│  │                                          │  │
│  │  ┌────────┐  ┌────────┐   ┌─────────┐  │  │
│  │  │ Linux  │  │Windows │   │   Mac   │  │  │
│  │  │        │  │        │   │         │  │  │
│  │  │ .so    │  │  .dll  │   │  .dylib │  │  │
│  │  └────────┘  └────────┘   └─────────┘  │  │
│  └──────────────────────────────────────────┘  │
│                      ↓                          │
│                  API Library                    │
│                (libcochl_api)                   │
└─────────────────────────────────────────────────┘
                      ↓
        ┌─────────────────────────────┐
        │         Kernel              │
        │  ┌────────┐ ┌─────────────┐ │
        │  │ Memory │ │Temperature  │ │
        │  └────────┘ └─────────────┘ │
        │  ┌────────┐                 │
        │  │ Delay  │                 │
        │  └────────┘                 │
        └─────────────────────────────┘
```

## Features

- **Multi-Runtime Support**: Automatically selects TFLite or LibTorch based on model extension
- **Cross-Platform**: Linux (x64, ARM64), Windows, macOS
- **Performance Profiling**: Built-in profiler for latency and throughput tracking
- **Async Inference**: Thread pool for parallel processing
- **System Monitoring**: Memory, CPU temperature, latency tracking
- **Dynamic Loading**: Uses API as shared library (no recompilation needed)

## Supported Platforms

| Platform | Architecture | Library |
|----------|--------------|---------|
| Linux    | x64          | libcochl_api.so |
| Linux    | ARM64 (Raspberry Pi 5) | libcochl_api.so |
| Windows  | x64          | cochl_api.dll |
| macOS    | x64/ARM64    | libcochl_api.dylib |

## Quick Start

### 1. Build API Library

First, build the API as a shared library:

```bash
cd api
mkdir build && cd build

# Linux/Mac
cmake -DCMAKE_BUILD_TYPE=Release \
      -DUSE_TFLITE=ON \
      -DUSE_LIBTORCH=ON \
      -DBUILD_SHARED_LIBS=ON \
      ..
make -j$(nproc)

# This creates libcochl_api.so (Linux) or libcochl_api.dylib (macOS)
```

### 2. Build SDK

```bash
cd sdk
mkdir build && cd build

cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### 3. Deploy to Edge Device

Copy the following to your edge device:
- `libcochl_api.so` (API library)
- Your compiled SDK application
- Model file (`.tflite` or `.pt`)

### 4. Usage Example

```cpp
#include "inference_engine.h"
#include <iostream>
#include <vector>

int main() {
  // Create engine (automatically detects runtime from extension)
  auto engine = cochl_sdk::InferenceEngine::Create(
      "model.tflite",  // Model path
      4,               // Number of threads
      true             // Enable profiler
  );

  if (!engine) {
    std::cerr << "Failed to create engine" << std::endl;
    return 1;
  }

  // Prepare input
  size_t input_size = engine->GetInputSize();
  size_t output_size = engine->GetOutputSize();

  std::vector<float> input(input_size, 0.5f);
  std::vector<float> output(output_size);

  // Run inference
  if (engine->RunInference(input.data(), input_size,
                          output.data(), output_size)) {
    std::cout << "Inference succeeded!" << std::endl;
  }

  // Get profiling stats
  std::cout << engine->GetProfilingStats() << std::endl;

  // Get system resource usage
  std::cout << engine->GetResourceUsage() << std::endl;

  return 0;
}
```

### 5. Async Inference Example

```cpp
// Run inference asynchronously
engine->RunInferenceAsync(input.data(), input_size,
                         output.data(), output_size,
                         [](bool success) {
  if (success) {
    std::cout << "Async inference completed" << std::endl;
  }
});
```

## Model Format Support

| Extension | Runtime | Notes |
|-----------|---------|-------|
| `.tflite` | TensorFlow Lite | Optimized for mobile/edge |
| `.pt`, `.pth` | LibTorch | PyTorch TorchScript models |

## System Monitoring

The SDK provides kernel-level monitoring:

- **Memory**: Total, used, available memory and usage percentage
- **Temperature**: CPU temperature (Linux/Raspberry Pi only)
- **Latency**: Min/max/average inference latency

```cpp
std::string status = engine->GetResourceUsage();
std::cout << status << std::endl;

// Output:
// Memory: 1024 MB / 4096 MB (25%)
// CPU Temperature: 45.2 °C
// Latency: min=5.2 ms, max=12.1 ms, avg=7.8 ms (100 samples)
```

## Raspberry Pi 5 Deployment

### Prerequisites

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake
```

### Build for ARM64

```bash
# On Raspberry Pi or using cross-compilation
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Run

```bash
export LD_LIBRARY_PATH=./api/build:$LD_LIBRARY_PATH
./sdk_example model.tflite
```

## Performance Tips

1. **Thread Pool Size**: Set to number of CPU cores for best performance
2. **Profiler**: Disable in production for lower overhead
3. **Model Format**: Use TFLite for smaller models, LibTorch for larger ones
4. **Memory**: Monitor usage with `GetResourceUsage()` to avoid OOM

## Troubleshooting

### "Failed to load library: libcochl_api.so"

Set `LD_LIBRARY_PATH`:
```bash
export LD_LIBRARY_PATH=/path/to/api/library:$LD_LIBRARY_PATH
```

### "Unsupported model format"

Check file extension:
- TFLite: `.tflite`
- LibTorch: `.pt` or `.pth`

### Temperature monitoring not working

Temperature monitoring is only supported on Linux. On other platforms, it will show "Not supported".

## License

See LICENSE file for details.
