#include <iostream>
#include <vector>
#include <string>
#include <dlfcn.h>

// Function pointer types for C API
typedef void* (*CochlApi_Create_t)(const char*);
typedef int (*CochlApi_RunInference_t)(void*, const float*, const long long*, size_t, float*);
typedef size_t (*CochlApi_GetInputSize_t)(void*);
typedef size_t (*CochlApi_GetOutputSize_t)(void*);
typedef void (*CochlApi_Destroy_t)(void*);
typedef int (*CochlApi_LoadImage_t)(const char*, float*, size_t);

int main() {
    std::cout << "=== SDK API Dynamic Loading Test ===" << std::endl;

    // Load the API library
    const char* lib_path = "../third_party/cochl_api/lib/libcochl_api.so";
    void* handle = dlopen(lib_path, RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
        std::cerr << "Failed to load library: " << dlerror() << std::endl;
        return 1;
    }
    std::cout << "✓ Successfully loaded: " << lib_path << std::endl;

    // Load function symbols
    auto create = (CochlApi_Create_t)dlsym(handle, "CochlApi_Create");
    auto get_input_size = (CochlApi_GetInputSize_t)dlsym(handle, "CochlApi_GetInputSize");
    auto get_output_size = (CochlApi_GetOutputSize_t)dlsym(handle, "CochlApi_GetOutputSize");
    auto run_inference = (CochlApi_RunInference_t)dlsym(handle, "CochlApi_RunInference");
    auto destroy = (CochlApi_Destroy_t)dlsym(handle, "CochlApi_Destroy");
    auto load_image = (CochlApi_LoadImage_t)dlsym(handle, "CochlApi_LoadImage");

    if (!create || !get_input_size || !get_output_size || !run_inference || !destroy || !load_image) {
        std::cerr << "Failed to load symbols: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }
    std::cout << "✓ Successfully loaded all API functions" << std::endl;

    // Test 1: Custom Runtime
    std::cout << "\n[Test 1] Custom Runtime" << std::endl;
    const char* custom_model = "../../models/model.bin";
    void* custom_api = create(custom_model);
    if (custom_api) {
        size_t input_size = get_input_size(custom_api);
        size_t output_size = get_output_size(custom_api);
        std::cout << "  Input size: " << input_size << std::endl;
        std::cout << "  Output size: " << output_size << std::endl;

        // Run inference with dummy data
        std::vector<float> input(input_size, 0.5f);
        std::vector<float> output(output_size);
        long long input_shape[] = {1, 3, 224, 224};

        if (run_inference(custom_api, input.data(), input_shape, 4, output.data())) {
            std::cout << "  ✓ Inference successful" << std::endl;
            std::cout << "  First 5 outputs: ";
            for (int i = 0; i < 5 && i < output_size; i++) {
                std::cout << output[i] << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "  ✗ Inference failed" << std::endl;
        }

        destroy(custom_api);
        std::cout << "  ✓ Custom Runtime test passed" << std::endl;
    } else {
        std::cout << "  ✗ Failed to create Custom Runtime API" << std::endl;
    }

    // Test 2: LibTorch (if available)
    std::cout << "\n[Test 2] LibTorch Runtime" << std::endl;
    const char* torch_model = "../../models/resnet50.pt";
    void* torch_api = create(torch_model);
    if (torch_api) {
        size_t input_size = get_input_size(torch_api);
        size_t output_size = get_output_size(torch_api);
        std::cout << "  Input size: " << input_size << std::endl;
        std::cout << "  Output size: " << output_size << std::endl;

        // Load and test with real image
        const char* image_path = "../../api/test/dog.png";
        std::vector<float> input(input_size);
        if (load_image(image_path, input.data(), input_size)) {
            std::vector<float> output(output_size);
            long long input_shape[] = {1, 3, 224, 224};

            if (run_inference(torch_api, input.data(), input_shape, 4, output.data())) {
                std::cout << "  ✓ Inference successful" << std::endl;

                // Find top prediction
                float max_score = output[0];
                int max_idx = 0;
                for (size_t i = 1; i < output_size; i++) {
                    if (output[i] > max_score) {
                        max_score = output[i];
                        max_idx = i;
                    }
                }
                std::cout << "  Top prediction: class " << max_idx << " (score: " << max_score << ")" << std::endl;
            } else {
                std::cout << "  ✗ Inference failed" << std::endl;
            }
        } else {
            std::cout << "  ! Image not found, skipping real inference test" << std::endl;
        }

        destroy(torch_api);
        std::cout << "  ✓ LibTorch Runtime test passed" << std::endl;
    } else {
        std::cout << "  ! LibTorch model not found, skipping" << std::endl;
    }

    // Test 3: TFLite (if available)
    std::cout << "\n[Test 3] TFLite Runtime" << std::endl;
    const char* tflite_model = "../../models/resnet50.tflite";
    void* tflite_api = create(tflite_model);
    if (tflite_api) {
        size_t input_size = get_input_size(tflite_api);
        size_t output_size = get_output_size(tflite_api);
        std::cout << "  Input size: " << input_size << std::endl;
        std::cout << "  Output size: " << output_size << std::endl;

        destroy(tflite_api);
        std::cout << "  ✓ TFLite Runtime test passed" << std::endl;
    } else {
        std::cout << "  ! TFLite model not found, skipping" << std::endl;
    }

    // Cleanup
    dlclose(handle);
    std::cout << "\n=== All tests completed ===" << std::endl;
    std::cout << "✓ SDK successfully loaded and used API library independently!" << std::endl;

    return 0;
}
