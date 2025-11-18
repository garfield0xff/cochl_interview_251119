#include "cochl_api_c.h"

#include "cochl_api.h"

#define STB_IMAGE_IMPLEMENTATION
#include "utils/util_img.h"

#include <iostream>
#include <map>
#include <string>

extern "C" {

void* CochlApi_Create(const char* model_path) {
  if (!model_path) {
    std::cerr << "[CochlApi_Create] NULL model path" << std::endl;
    return nullptr;
  }

  auto api = external_api::CochlApi::Create(std::string(model_path));
  if (!api) {
    return nullptr;
  }

  // Release ownership and return raw pointer
  return api.release();
}

int CochlApi_RunInference(void* instance, const float* input, size_t input_size,
                          float* output, size_t output_size) {
  if (!instance) {
    std::cerr << "[CochlApi_RunInference] NULL instance" << std::endl;
    return 0;
  }

  auto* api = static_cast<external_api::CochlApi*>(instance);
  return api->RunInference(input, input_size, output, output_size) ? 1 : 0;
}

size_t CochlApi_GetInputSize(void* instance) {
  if (!instance) {
    return 0;
  }

  auto* api = static_cast<external_api::CochlApi*>(instance);
  return api->GetInputSize();
}

size_t CochlApi_GetOutputSize(void* instance) {
  if (!instance) {
    return 0;
  }

  auto* api = static_cast<external_api::CochlApi*>(instance);
  return api->GetOutputSize();
}

void CochlApi_Destroy(void* instance) {
  if (!instance) {
    return;
  }

  auto* api = static_cast<external_api::CochlApi*>(instance);
  delete api;
}

int CochlApi_LoadImage(const char* image_path, float* output_data, size_t output_size) {
  if (!image_path || !output_data || output_size == 0) {
    std::cerr << "[CochlApi_LoadImage] Invalid parameters" << std::endl;
    return 0;
  }

  // Load and preprocess image
  auto input_hwc = cochl_api::utils::LoadAndPreprocessImage(std::string(image_path));
  if (input_hwc.empty()) {
    std::cerr << "[CochlApi_LoadImage] Failed to load image: " << image_path << std::endl;
    return 0;
  }

  // Convert HWC to NCHW
  auto input_nchw = cochl_api::utils::HWCToNCHW(input_hwc, 224, 224, 3);

  if (input_nchw.size() != output_size) {
    std::cerr << "[CochlApi_LoadImage] Size mismatch. Expected: " << output_size
              << ", Got: " << input_nchw.size() << std::endl;
    return 0;
  }

  // Copy to output buffer
  std::copy(input_nchw.begin(), input_nchw.end(), output_data);
  return 1;
}

void* CochlApi_LoadClassNames(const char* json_path) {
  if (!json_path) {
    std::cerr << "[CochlApi_LoadClassNames] NULL json path" << std::endl;
    return nullptr;
  }

  auto class_map = cochl_api::utils::LoadImageNetClasses(std::string(json_path));
  if (class_map.empty()) {
    std::cerr << "[CochlApi_LoadClassNames] Failed to load class names" << std::endl;
    return nullptr;
  }

  // Allocate and return map pointer
  auto* map_ptr = new std::map<int, std::string>(std::move(class_map));
  return map_ptr;
}

const char* CochlApi_GetClassName(void* class_map, int class_idx) {
  if (!class_map) {
    return nullptr;
  }

  auto* map_ptr = static_cast<std::map<int, std::string>*>(class_map);
  auto it = map_ptr->find(class_idx);
  if (it != map_ptr->end()) {
    return it->second.c_str();
  }

  return nullptr;
}

void CochlApi_DestroyClassMap(void* class_map) {
  if (!class_map) {
    return;
  }

  auto* map_ptr = static_cast<std::map<int, std::string>*>(class_map);
  delete map_ptr;
}

}  // extern "C"
