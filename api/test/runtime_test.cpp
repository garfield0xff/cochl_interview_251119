#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "runtime/runtime_manager.h"
#include "utils/util_img.h"

namespace cochl_api {
namespace runtime {
namespace test {

class RuntimeTest : public ::testing::Test {
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
}  // namespace runtime
}  // namespace cochl_api


using cochl_api::runtime::test::RuntimeTest;
using cochl_api::runtime::RuntimeManager;
using cochl_api::utils::LoadImageNetClasses;
using cochl_api::utils::LoadAndPreprocessImage;
using cochl_api::utils::HWCToNCHW;
using cochl_api::utils::GetTopKPredictions;
using cochl_api::utils::GetClassName;

// Test runtime switching
TEST_F(RuntimeTest, RuntimeSwitching) {
#ifdef USE_TFLITE
  const std::string tflite_path = std::string(PROJECT_ROOT) + "/models/resnet50.tflite";
  if (FileExists(tflite_path)) {
    auto tflite_manager = RuntimeManager::Create(tflite_path);
    ASSERT_NE(tflite_manager, nullptr);
    EXPECT_EQ(tflite_manager->GetInferenceEngineType(), RuntimeManager::InferenceEngine::TFLITE);
    std::cout << "[RuntimeSwitching] TFLite runtime loaded successfully" << std::endl;
  }
#endif

#ifdef USE_LIBTORCH
  const std::string torch_path = std::string(PROJECT_ROOT) + "/models/resnet50.pt";
  if (FileExists(torch_path)) {
    auto torch_manager = RuntimeManager::Create(torch_path);
    ASSERT_NE(torch_manager, nullptr);
    EXPECT_EQ(torch_manager->GetInferenceEngineType(), RuntimeManager::InferenceEngine::LIBTORCH);
    std::cout << "[RuntimeSwitching] LibTorch runtime loaded successfully" << std::endl;
  }
#endif
}

#ifdef USE_TFLITE
// Test inference Dog IMG
TEST_F(RuntimeTest, TFLiteResNet50) {
  const std::string model_path = std::string(PROJECT_ROOT) + "/models/resnet50.tflite";
  const std::string image_path = std::string(PROJECT_ROOT) + "/api/test/dog.png";
  const std::string class_json = std::string(PROJECT_ROOT) + "/api/test/imagenet_class_index.json";

  if (!FileExists(model_path)) {
    GTEST_SKIP() << "ResNet50 TFLite model not found at: " << model_path;
  }

  if (!FileExists(image_path)) {
    GTEST_SKIP() << "Test image not found at: " << image_path;
  }

  // Load ImageNet class names
  auto class_map = LoadImageNetClasses(class_json);

  auto manager = RuntimeManager::Create(model_path);
  ASSERT_NE(manager, nullptr) << "Failed to load TFLite ResNet50 model";

  EXPECT_EQ(manager->GetInferenceEngineType(), RuntimeManager::InferenceEngine::TFLITE);
  EXPECT_GT(manager->GetInputSize(), 0);
  EXPECT_GT(manager->GetOutputSize(), 0);

  std::cout << "\n[TFLite ResNet50] Input size: " << manager->GetInputSize() << std::endl;
  std::cout << "[TFLite ResNet50] Output size: " << manager->GetOutputSize() << std::endl;

  // Load img
  auto input_hwc = LoadAndPreprocessImage(image_path);
  ASSERT_FALSE(input_hwc.empty()) << "Failed to load image: " << image_path;

  // Convert HWC to NCHW
  auto input = HWCToNCHW(input_hwc, 224, 224, 3);
  ASSERT_EQ(input.size(), manager->GetInputSize())
      << "Image size mismatch. Expected: " << manager->GetInputSize() << ", Got: " << input.size();

  // Run inference
  std::vector<float> output(manager->GetOutputSize());
  ASSERT_TRUE(manager->RunInference(input.data(), input.size(), output.data(), output.size()))
      << "TFLite inference failed";

  auto top5 = GetTopKPredictions(output, 5);
  std::cout << "\n[TFLite ResNet50] Top 5 predictions for dog.png:" << std::endl;
  for (const auto& [class_idx, score] : top5) {
    std::string class_name = GetClassName(class_map, class_idx);
    std::cout << "  " << class_idx << ": " << class_name << " (" << score << ")" << std::endl;
  }
}

#endif

#ifdef USE_LIBTORCH
// Test inference Dog IMG for LIBTORCH
TEST_F(RuntimeTest, LibTorchResNet50) {
  const std::string model_path = std::string(PROJECT_ROOT) + "/models/resnet50.pt";
  const std::string image_path = std::string(PROJECT_ROOT) + "/api/test/dog.png";
  const std::string class_json = std::string(PROJECT_ROOT) + "/api/test/imagenet_class_index.json";

  if (!FileExists(model_path)) {
    GTEST_SKIP() << "ResNet50 PyTorch model not found at: " << model_path;
  }

  if (!FileExists(image_path)) {
    GTEST_SKIP() << "Test image not found at: " << image_path;
  }

  auto class_map = LoadImageNetClasses(class_json);

  auto manager = RuntimeManager::Create(model_path);
  ASSERT_NE(manager, nullptr) << "Failed to load LibTorch ResNet50 model";

  EXPECT_EQ(manager->GetInferenceEngineType(), RuntimeManager::InferenceEngine::LIBTORCH);
  EXPECT_GT(manager->GetInputSize(), 0);
  EXPECT_GT(manager->GetOutputSize(), 0);

  std::cout << "\n[LibTorch ResNet50] Input size: " << manager->GetInputSize() << std::endl;
  std::cout << "[LibTorch ResNet50] Output size: " << manager->GetOutputSize() << std::endl;

  auto input_hwc = LoadAndPreprocessImage(image_path);
  ASSERT_FALSE(input_hwc.empty()) << "Failed to load image: " << image_path;

  auto input = HWCToNCHW(input_hwc, 224, 224, 3);
  ASSERT_EQ(input.size(), manager->GetInputSize())
      << "Image size mismatch. Expected: " << manager->GetInputSize() << ", Got: " << input.size();

  std::vector<float> output(manager->GetOutputSize());
  ASSERT_TRUE(manager->RunInference(input.data(), input.size(), output.data(), output.size()))
      << "LibTorch inference failed";

  auto top5 = GetTopKPredictions(output, 5);
  std::cout << "\n[LibTorch ResNet50] Top 5 predictions for dog.png:" << std::endl;
  for (const auto& [class_idx, score] : top5) {
    std::string class_name = GetClassName(class_map, class_idx);
    std::cout << "  " << class_idx << ": " << class_name << " (" << score << ")" << std::endl;
  }
}
#endif

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
