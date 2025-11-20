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
 * @brief Tensor layout types
 */
#define TENSOR_LAYOUT_NCHW 0  // Batch, Channel, Height, Width (PyTorch, Caffe)
#define TENSOR_LAYOUT_NHWC 1  // Batch, Height, Width, Channel (TensorFlow)

/**
 * @brief Run inference
 * @param instance CochlApi instance
 * @param input Input data array
 * @param input_shape Shape of input tensor (e.g., [1, 3, 224, 224])
 * @param shape_size Number of dimensions in input_shape
 * @param output Output data array (must be pre-allocated with CochlApi_GetOutputSize())
 * @param layout Tensor layout (0=NCHW, 1=NHWC)
 * @return 1 if successful, 0 otherwise
 */
int CochlApi_RunInference(void* instance, const float* input,
                          const long long* input_shape, size_t shape_size,
                          float* output, int layout);

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

/**
 * @brief Load and preprocess image for ResNet50
 * @param image_path Path to image file
 * @param output_data Pre-allocated buffer for output (should be 150528 floats for 224x224x3)
 * @param output_size Size of output buffer
 * @return 1 if successful, 0 otherwise
 */
int CochlApi_LoadImage(const char* image_path, float* output_data, size_t output_size);

/**
 * @brief Load ImageNet class names from JSON file
 * @param json_path Path to imagenet_class_index.json
 * @return Opaque pointer to class map, NULL on failure
 */
void* CochlApi_LoadClassNames(const char* json_path);

/**
 * @brief Get class name from index
 * @param class_map Class map instance
 * @param class_idx Class index
 * @return Class name string (do not free), NULL if not found
 */
const char* CochlApi_GetClassName(void* class_map, int class_idx);

/**
 * @brief Destroy class map
 * @param class_map Class map instance
 */
void CochlApi_DestroyClassMap(void* class_map);

#ifdef __cplusplus
}
#endif

#endif  // COCHL_API_C_H
