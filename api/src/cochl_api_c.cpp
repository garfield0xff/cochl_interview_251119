#include "cochl_api_c.h"

#include "cochl_api.h"

#include <iostream>

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

}  // extern "C"
