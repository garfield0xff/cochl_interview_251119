#ifndef COCHL_API_C_H
#define COCHL_API_C_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief C API for CochlApi (for dynamic library loading)
 *
 * Provides C interface that can be loaded dynamically by SDK.
 */

/**
 * @brief Create CochlApi instance
 * @param model_path Path to model file
 * @return Opaque pointer to CochlApi instance, NULL on failure
 */
void* CochlApi_Create(const char* model_path);

/**
 * @brief Run inference
 * @param instance CochlApi instance
 * @param input Input data array
 * @param input_size Size of input array
 * @param output Output data array
 * @param output_size Size of output array
 * @return 1 if successful, 0 otherwise
 */
int CochlApi_RunInference(void* instance, const float* input, size_t input_size,
                          float* output, size_t output_size);

/**
 * @brief Get input size required by model
 * @param instance CochlApi instance
 * @return Input size
 */
size_t CochlApi_GetInputSize(void* instance);

/**
 * @brief Get output size produced by model
 * @param instance CochlApi instance
 * @return Output size
 */
size_t CochlApi_GetOutputSize(void* instance);

/**
 * @brief Destroy CochlApi instance
 * @param instance CochlApi instance
 */
void CochlApi_Destroy(void* instance);

#ifdef __cplusplus
}
#endif

#endif  // COCHL_API_C_H
