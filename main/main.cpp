#include <glog/logging.h>
#include <iostream>
#include <vector>
#include <algorithm>

#include "inference_engine.h"

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;  
    // Parse arguments
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << "<model> <image> <class_json>" << std::endl;
        std::cerr << "Example: " << argv[0]
                  << " ./models/resnet50.tflite"
                  << " ./api/test/dog.png"
                  << " ./api/test/imagenet_class_index.json" << std::endl;
        return 1;
    }

    std::string library_path = COCHL_API_LIB_PATH;
    std::string model_path = argv[1];
    std::string image_path = argv[2];
    std::string class_json = argv[3];

    cochl::InferenceEngine engine;

    std::cout << "\n[1] Loading library: " << library_path << std::endl;
    if (!engine.loadLibrary(library_path)) {
        std::cerr << "Failed to load library" << std::endl;
        return 1;
    }

    std::cout << "\n[2] Creating API instance with model: " << model_path << std::endl;
    if (!engine.create(model_path)) {
        std::cerr << "Failed to create API instance" << std::endl;
        return 1;
    }

    // Load class names
    std::cout << "\n[3] Loading ImageNet class names: " << class_json << std::endl;
    if (!engine.loadClassNames(class_json)) {
        std::cerr << "Failed to load class names" << std::endl;
        return 1;
    }

    // Define input shape (NCHW: 1 batch, 3 channels, 224x224)
    std::vector<int64_t> input_shape = {1, 3, 224, 224};

    // Calculate input size from shape
    size_t input_size = 1;
    for (auto dim : input_shape) {
        input_size *= dim;
    }

    // Get output size from model
    size_t output_size = engine.getOutputSize();

    std::cout << "\n[4] Model information:" << std::endl;
    std::cout << "  Input shape: [";
    for (size_t i = 0; i < input_shape.size(); ++i) {
        std::cout << input_shape[i];
        if (i < input_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  Input size: " << input_size << std::endl;
    std::cout << "  Output size: " << output_size << std::endl;

    // Load and preprocess image
    std::cout << "\n[5] Loading and preprocessing image: " << image_path << std::endl;
    std::vector<float> input(input_size);
    if (!engine.loadImage(image_path, input.data(), input.size())) {
        std::cerr << "Failed to load image" << std::endl;
        return 1;
    }

    std::vector<float> output(output_size);

    // Run inference
    std::cout << "\n[6] Running inference..." << std::endl;
    if (!engine.runInference(input.data(), input_shape, output.data())) {
        std::cerr << "Inference failed" << std::endl;
        return 1;
    }

    // Display top-5 results with class names
    std::cout << "\n[7] Top 5 predictions:" << std::endl;

    // Create index-score pairs
    std::vector<std::pair<int, float>> indexed_output;
    for (size_t i = 0; i < output.size(); ++i) {
        indexed_output.push_back({static_cast<int>(i), output[i]});
    }

    // Sort by score descending
    std::partial_sort(indexed_output.begin(),
                      indexed_output.begin() + std::min(5, static_cast<int>(indexed_output.size())),
                      indexed_output.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });

    for (int i = 0; i < std::min(5, static_cast<int>(indexed_output.size())); ++i) {
        int class_idx = indexed_output[i].first;
        float score = indexed_output[i].second;
        std::string class_name = engine.getClassName(class_idx);

        std::cout << "  " << (i + 1) << ". " << class_name
                  << " (class " << class_idx << "): " << score << std::endl;
    }

    return 0;
}
