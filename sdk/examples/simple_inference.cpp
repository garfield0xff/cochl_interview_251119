#include "inference_engine.h"

#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
    std::cerr << "Example: " << argv[0] << " model.tflite" << std::endl;
    return 1;
  }

  const char* model_path = argv[1];

  std::cout << "=== Cochl Edge SDK Example ===" << std::endl;
  std::cout << "Model: " << model_path << std::endl;

  // Create inference engine
  auto engine = cochl_sdk::InferenceEngine::Create(
      model_path,
      0,     // Auto-detect thread count
      true   // Enable profiler
  );

  if (!engine) {
    std::cerr << "Failed to create inference engine" << std::endl;
    return 1;
  }

  std::cout << "\nEngine created successfully!" << std::endl;
  std::cout << "Input size: " << engine->GetInputSize() << std::endl;
  std::cout << "Output size: " << engine->GetOutputSize() << std::endl;

  // Prepare test input (dummy data)
  size_t input_size = engine->GetInputSize();
  size_t output_size = engine->GetOutputSize();

  std::vector<float> input(input_size);
  std::vector<float> output(output_size);

  // Fill input with test pattern
  for (size_t i = 0; i < input_size; ++i) {
    input[i] = static_cast<float>(i % 100) / 100.0f;
  }

  std::cout << "\n=== Running Inference ===" << std::endl;

  // Run inference 10 times for profiling
  for (int i = 0; i < 10; ++i) {
    bool success = engine->RunInference(input.data(), input_size,
                                       output.data(), output_size);

    if (!success) {
      std::cerr << "Inference " << i + 1 << " failed" << std::endl;
      return 1;
    }

    std::cout << "Inference " << i + 1 << " completed" << std::endl;
  }

  // Print first few output values
  std::cout << "\n=== Output (first 5 values) ===" << std::endl;
  for (size_t i = 0; i < std::min(size_t(5), output_size); ++i) {
    std::cout << "output[" << i << "] = " << output[i] << std::endl;
  }

  // Print profiling statistics
  std::cout << "\n" << engine->GetProfilingStats() << std::endl;

  // Print system resource usage
  std::cout << "\n=== System Resources ===" << std::endl;
  std::cout << engine->GetResourceUsage() << std::endl;

  return 0;
}
