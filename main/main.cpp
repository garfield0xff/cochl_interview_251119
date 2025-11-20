#include <iostream>
#include <vector>
#include <algorithm>

#include "inference_engine.h"

int main(int argc, char** argv) {
    std::cout << "=== Cochl Inference Engine Test ===" << std::endl;

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

    std::cout << "\n[2] Loading model: " << model_path << std::endl;
    if (!engine.loadModel(model_path)) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    // Get input/output sizes
    size_t input_size = engine.getInputSize();
    size_t output_size = engine.getOutputSize();

    std::cout << "\n[3] Model information:" << std::endl;
    std::cout << "  Input size: " << input_size << std::endl;
    std::cout << "  Output size: " << output_size << std::endl;

    // Load class names
    std::cout << "\n[4] Loading ImageNet class names: " << class_json << std::endl;
    if (!engine.loadClassNames(class_json)) {
        std::cerr << "Failed to load class names" << std::endl;
        return 1;
    }

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
    auto status = engine.runInference(input.data(), input.size(),
                                       output.data(), output.size());

    if (status != cochl::InferenceStatus::OK) {
        std::cerr << "Inference failed with status: " << static_cast<int>(status) << std::endl;
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

    std::cout << "\n=== Test completed successfully ===" << std::endl;
    return 0;
}
