#include <gtest/gtest.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

#include "cochl_api_c.h"

namespace cochl_api {
namespace test {

class ApiTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}

  // create dummy data for input sample data
  std::vector<float> CreateDummyInput(size_t size) {
    std::vector<float> input(size);
    for (size_t i = 0; i < size; ++i) {
      input[i] = static_cast<float>(i % 256) / 255.0f;
    }
    return input;
  }
  bool FileExists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
  }
};

}  // namespace test
}  // namespace cochl_api


using cochl_api::test::ApiTest;

/**
 * =================================================================
 *   Initialzie Api
 * =================================================================
 */
TEST_F(ApiTest, ApiInitialize) {
#ifdef USE_TFLITE
  const std::string tflite_path = std::string(PROJECT_ROOT) + "/models/resnet50.tflite";
  if (FileExists(tflite_path)) {
    void* api = CochlApi_Create(tflite_path.c_str());
    ASSERT_NE(api, nullptr);
    CochlApi_Destroy(api);
  }
#endif

#ifdef USE_TVM
  const std::string tvm_path = std::string(PROJECT_ROOT) + "/models/resnet50_tvm.so";
  if (FileExists(tvm_path)) {
    void* api = CochlApi_Create(tvm_path.c_str());
    ASSERT_NE(api, nullptr);
    EXPECT_GT(CochlApi_GetInputSize(api), 0);
    EXPECT_GT(CochlApi_GetOutputSize(api), 0);
    std::cout << "[ApiCreation] TVM API created successfully" << std::endl;
    std::cout << "[ApiCreation] Input size: " << CochlApi_GetInputSize(api) << std::endl;
    std::cout << "[ApiCreation] Output size: " << CochlApi_GetOutputSize(api) << std::endl;
    CochlApi_Destroy(api);
  }
#endif

#ifdef USE_LIBTORCH
  const std::string torch_path = std::string(PROJECT_ROOT) + "/models/resnet50.pt";
  if (FileExists(torch_path)) {
    void* api = CochlApi_Create(torch_path.c_str());
    ASSERT_NE(api, nullptr);
    EXPECT_GT(CochlApi_GetInputSize(api), 0);
    EXPECT_GT(CochlApi_GetOutputSize(api), 0);
    std::cout << "[ApiCreation] LibTorch API created successfully" << std::endl;
    std::cout << "[ApiCreation] Input size: " << CochlApi_GetInputSize(api) << std::endl;
    std::cout << "[ApiCreation] Output size: " << CochlApi_GetOutputSize(api) << std::endl;
    CochlApi_Destroy(api);
  }
#endif

#ifdef USE_CUSTOM
  const std::string custom_path = std::string(PROJECT_ROOT) + "/models/model.bin";
  void* api = CochlApi_Create(custom_path.c_str());
  ASSERT_NE(api, nullptr);
  EXPECT_GT(CochlApi_GetInputSize(api), 0);
  EXPECT_GT(CochlApi_GetOutputSize(api), 0);
  std::cout << "[ApiCreation] Custom runtime API created successfully" << std::endl;
  std::cout << "[ApiCreation] Input size: " << CochlApi_GetInputSize(api) << std::endl;
  std::cout << "[ApiCreation] Output size: " << CochlApi_GetOutputSize(api) << std::endl;
  CochlApi_Destroy(api);
#endif
}




/**
 * =================================================================
 *   Test Runtime ( tflite, libtorch, Custom )
 * =================================================================
 */
#ifdef USE_TFLITE
TEST_F(ApiTest, TFLiteResNet50) {
  const std::string model_path = std::string(PROJECT_ROOT) + "/models/resnet50.tflite";
  const std::string image_path = std::string(PROJECT_ROOT) + "/api/test/dog.png";
  const std::string class_json = std::string(PROJECT_ROOT) + "/api/test/imagenet_class_index.json";

  if (!FileExists(model_path)) {
    GTEST_SKIP() << "ResNet50 TFLite model not found at: " << model_path;
  }

  if (!FileExists(image_path)) {
    GTEST_SKIP() << "Test image not found at: " << image_path;
  }

  // Create API instance
  void* api = CochlApi_Create(model_path.c_str());
  ASSERT_NE(api, nullptr) << "Failed to create CochlApi instance";

  size_t input_size = CochlApi_GetInputSize(api);
  size_t output_size = CochlApi_GetOutputSize(api);

  EXPECT_GT(input_size, 0);
  EXPECT_GT(output_size, 0);

  std::cout << "\n[TFLite ResNet50 C API] Input size: " << input_size << std::endl;
  std::cout << "[TFLite ResNet50 C API] Output size: " << output_size << std::endl;

  // Load and preprocess image
  std::vector<float> input(input_size);
  int load_result = CochlApi_LoadImage(image_path.c_str(), input.data(), input_size);
  ASSERT_EQ(load_result, 1) << "Failed to load image: " << image_path;

  // Run inference
  std::vector<float> output(output_size);
  long long input_shape[] = {1, 3, 224, 224};  // NCHW format
  int inference_result = CochlApi_RunInference(api, input.data(), input_shape, 4, output.data());
  ASSERT_EQ(inference_result, 1) << "TFLite inference failed";

  // Load class names and get top 5 predictions
  void* class_map = CochlApi_LoadClassNames(class_json.c_str());
  if (class_map != nullptr) {
    // Find top 5 predictions
    std::vector<std::pair<int, float>> top5;
    for (size_t i = 0; i < output_size; ++i) {
      top5.push_back({static_cast<int>(i), output[i]});
    }
    std::partial_sort(top5.begin(), top5.begin() + 5, top5.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
    top5.resize(5);

    std::cout << "\n[TFLite ResNet50 C API] Top 5 predictions for dog.png:" << std::endl;
    for (const auto& [class_idx, score] : top5) {
      const char* class_name = CochlApi_GetClassName(class_map, class_idx);
      if (class_name != nullptr) {
        std::cout << "  " << class_idx << ": " << class_name << " (" << score << ")" << std::endl;
      }
    }

    CochlApi_DestroyClassMap(class_map);
  }

  CochlApi_Destroy(api);
}

#endif

#ifdef USE_LIBTORCH
// Test inference Dog IMG for LIBTORCH using C API
TEST_F(ApiTest, LibTorchResNet50) {
  const std::string model_path = std::string(PROJECT_ROOT) + "/models/resnet50.pt";
  const std::string image_path = std::string(PROJECT_ROOT) + "/api/test/dog.png";
  const std::string class_json = std::string(PROJECT_ROOT) + "/api/test/imagenet_class_index.json";

  if (!FileExists(model_path)) {
    GTEST_SKIP() << "ResNet50 PyTorch model not found at: " << model_path;
  }

  if (!FileExists(image_path)) {
    GTEST_SKIP() << "Test image not found at: " << image_path;
  }

  // Create API instance
  void* api = CochlApi_Create(model_path.c_str());
  ASSERT_NE(api, nullptr) << "Failed to create CochlApi instance";

  size_t input_size = CochlApi_GetInputSize(api);
  size_t output_size = CochlApi_GetOutputSize(api);

  EXPECT_GT(input_size, 0);
  EXPECT_GT(output_size, 0);

  std::cout << "\n[LibTorch ResNet50 C API] Input size: " << input_size << std::endl;
  std::cout << "[LibTorch ResNet50 C API] Output size: " << output_size << std::endl;

  // Load and preprocess image
  std::vector<float> input(input_size);
  int load_result = CochlApi_LoadImage(image_path.c_str(), input.data(), input_size);
  ASSERT_EQ(load_result, 1) << "Failed to load image: " << image_path;

  // Run inference
  std::vector<float> output(output_size);
  long long input_shape[] = {1, 3, 224, 224};  // NCHW format
  int inference_result = CochlApi_RunInference(api, input.data(), input_shape, 4, output.data());
  ASSERT_EQ(inference_result, 1) << "LibTorch inference failed";

  // Load class names and get top 5 predictions
  void* class_map = CochlApi_LoadClassNames(class_json.c_str());
  if (class_map != nullptr) {
    // Find top 5 predictions
    std::vector<std::pair<int, float>> top5;
    for (size_t i = 0; i < output_size; ++i) {
      top5.push_back({static_cast<int>(i), output[i]});
    }
    std::partial_sort(top5.begin(), top5.begin() + 5, top5.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
    top5.resize(5);

    std::cout << "\n[LibTorch ResNet50 C API] Top 5 predictions for dog.png:" << std::endl;
    for (const auto& [class_idx, score] : top5) {
      const char* class_name = CochlApi_GetClassName(class_map, class_idx);
      if (class_name != nullptr) {
        std::cout << "  " << class_idx << ": " << class_name << " (" << score << ")" << std::endl;
      }
    }

    CochlApi_DestroyClassMap(class_map);
  }

  CochlApi_Destroy(api);
}
#endif

#ifdef USE_TVM
// Test TVM Runtime with ResNet50
TEST_F(ApiTest, TVMResNet50) {
  const std::string model_path = std::string(PROJECT_ROOT) + "/models/resnet50_tvm.so";
  const std::string image_path = std::string(PROJECT_ROOT) + "/api/test/dog.png";
  const std::string class_json = std::string(PROJECT_ROOT) + "/api/test/imagenet_class_index.json";

  if (!FileExists(model_path)) {
    GTEST_SKIP() << "ResNet50 TVM model not found at: " << model_path;
  }

  if (!FileExists(image_path)) {
    GTEST_SKIP() << "Test image not found at: " << image_path;
  }

  // Create API instance
  void* api = CochlApi_Create(model_path.c_str());
  ASSERT_NE(api, nullptr) << "Failed to create CochlApi instance";

  size_t input_size = CochlApi_GetInputSize(api);
  size_t output_size = CochlApi_GetOutputSize(api);

  EXPECT_GT(input_size, 0);
  EXPECT_GT(output_size, 0);

  std::cout << "\n[TVM ResNet50 C API] Input size: " << input_size << std::endl;
  std::cout << "[TVM ResNet50 C API] Output size: " << output_size << std::endl;

  // Load and preprocess image
  std::vector<float> input(input_size);
  int load_result = CochlApi_LoadImage(image_path.c_str(), input.data(), input_size);
  ASSERT_EQ(load_result, 1) << "Failed to load image: " << image_path;

  // Run inference
  // Note: TVM model was converted from TFLite which uses NHWC format
  // The input shape should match what the model expects
  std::vector<float> output(output_size);
  long long input_shape[] = {1, 224, 224, 3};  // NHWC format (same as original TFLite)
  int inference_result = CochlApi_RunInference(api, input.data(), input_shape, 4, output.data());
  ASSERT_EQ(inference_result, 1) << "TVM inference failed";

  // Load class names and get top 5 predictions
  void* class_map = CochlApi_LoadClassNames(class_json.c_str());
  if (class_map != nullptr) {
    // Find top 5 predictions
    std::vector<std::pair<int, float>> top5;
    for (size_t i = 0; i < output_size; ++i) {
      top5.push_back({static_cast<int>(i), output[i]});
    }
    std::partial_sort(top5.begin(), top5.begin() + 5, top5.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
    top5.resize(5);

    std::cout << "\n[TVM ResNet50 C API] Top 5 predictions for dog.png:" << std::endl;
    for (const auto& [class_idx, score] : top5) {
      const char* class_name = CochlApi_GetClassName(class_map, class_idx);
      if (class_name != nullptr) {
        std::cout << "  " << class_idx << ": " << class_name << " (" << score << ")" << std::endl;
      }
    }

    CochlApi_DestroyClassMap(class_map);
  }

  CochlApi_Destroy(api);
}
#endif

#ifdef USE_CUSTOM
// Test Custom Runtime with mock inference
TEST_F(ApiTest, CustomRuntimeResNet50) {
  const std::string model_path = std::string(PROJECT_ROOT) + "/models/model.bin";
  const std::string image_path = std::string(PROJECT_ROOT) + "/api/test/dog.png";

  if (!FileExists(image_path)) {
    GTEST_SKIP() << "Test image not found at: " << image_path;
  }

  // Create API instance
  void* api = CochlApi_Create(model_path.c_str());
  ASSERT_NE(api, nullptr) << "Failed to create CochlApi instance";

  size_t input_size = CochlApi_GetInputSize(api);
  size_t output_size = CochlApi_GetOutputSize(api);

  EXPECT_GT(input_size, 0);
  EXPECT_GT(output_size, 0);

  std::cout << "\n[Custom Runtime C API] Input size: " << input_size << std::endl;
  std::cout << "[Custom Runtime C API] Output size: " << output_size << std::endl;

  // Load and preprocess image
  std::vector<float> input(input_size);
  int load_result = CochlApi_LoadImage(image_path.c_str(), input.data(), input_size);
  ASSERT_EQ(load_result, 1) << "Failed to load image: " << image_path;

  // Run inference
  std::vector<float> output(output_size);
  long long input_shape[] = {1, 3, 224, 224};  // NCHW format
  int inference_result = CochlApi_RunInference(api, input.data(), input_shape, 4, output.data());
  ASSERT_EQ(inference_result, 1) << "Custom runtime inference failed";

  // Find top 5 predictions (mock output from custom runtime)
  std::vector<std::pair<int, float>> top5;
  for (size_t i = 0; i < output_size; ++i) {
    top5.push_back({static_cast<int>(i), output[i]});
  }
  std::partial_sort(top5.begin(), top5.begin() + 5, top5.end(),
                    [](const auto& a, const auto& b) { return a.second > b.second; });
  top5.resize(5);

  std::cout << "\n[Custom Runtime C API] Top 5 predictions (mock output):" << std::endl;
  for (const auto& [class_idx, score] : top5) {
    std::cout << "  " << class_idx << ": " << score << std::endl;
  }

  CochlApi_Destroy(api);
}
#endif

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
