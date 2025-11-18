#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "stb_image.h"

namespace cochl_api {
namespace utils {

// ImageNet normalization parameters
constexpr float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
constexpr float IMAGENET_STD[3] = {0.229f, 0.224f, 0.225f};

/**
 * Load and preprocess image for ResNet50 inference
 * - Loads image using stb_image (supports PNG, JPG, BMP, etc.)
 * - Resizes to 224x224 using simple bilinear interpolation
 * - Normalizes with ImageNet mean and std
 *
 * @param image_path Path to the image file
 * @return Preprocessed image as float vector (224*224*3), empty if failed
 */
inline std::vector<float> LoadAndPreprocessImage(const std::string& image_path) {
  int width, height, channels;
  unsigned char* img = stbi_load(image_path.c_str(), &width, &height, &channels, 3);

  if (!img) {
    std::cerr << "[ImageLoader] Failed to load image: " << image_path << std::endl;
    std::cerr << "[ImageLoader] STB Error: " << stbi_failure_reason() << std::endl;
    return {};
  }

  std::cout << "[ImageLoader] Loaded image: " << image_path
            << " [" << width << "x" << height << "x3]" << std::endl;

  // Resize to 224x224 (simple nearest neighbor)
  const int target_size = 224;
  std::vector<float> resized(target_size * target_size * 3);

  for (int y = 0; y < target_size; ++y) {
    for (int x = 0; x < target_size; ++x) {
      int src_x = std::min(x * width / target_size, width - 1);
      int src_y = std::min(y * height / target_size, height - 1);

      for (int c = 0; c < 3; ++c) {
        int src_idx = (src_y * width + src_x) * 3 + c;
        int dst_idx = (y * target_size + x) * 3 + c;
        resized[dst_idx] = static_cast<float>(img[src_idx]) / 255.0f;
      }
    }
  }

  stbi_image_free(img);

  // Apply ImageNet normalization
  for (int i = 0; i < target_size * target_size; ++i) {
    for (int c = 0; c < 3; ++c) {
      int idx = i * 3 + c;
      resized[idx] = (resized[idx] - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
    }
  }

  std::cout << "[ImageLoader] Preprocessed to 224x224 with ImageNet normalization" << std::endl;
  return resized;
}

/**
 * Convert HWC (Height, Width, Channel) format to NCHW (Batch, Channel, Height, Width) format
 * Input:  [H, W, C] = [224, 224, 3]
 * Output: [N, C, H, W] = [1, 3, 224, 224]
 *
 * @param hwc_data Input data in HWC format
 * @param height Image height
 * @param width Image width
 * @param channels Number of channels
 * @return Data in NCHW format
 */
inline std::vector<float> HWCToNCHW(const std::vector<float>& hwc_data,
                                     int height, int width, int channels) {
  std::vector<float> nchw_data(hwc_data.size());

  for (int c = 0; c < channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        int hwc_idx = (h * width + w) * channels + c;
        int nchw_idx = c * (height * width) + h * width + w;
        nchw_data[nchw_idx] = hwc_data[hwc_idx];
      }
    }
  }

  return nchw_data;
}

/**
 * Get top-k predictions from model output
 *
 * @param output Model output (logits or probabilities)
 * @param k Number of top predictions to return
 * @return Vector of (class_index, score) pairs, sorted by score descending
 */
inline std::vector<std::pair<int, float>> GetTopKPredictions(
    const std::vector<float>& output, int k = 5) {
  std::vector<std::pair<int, float>> indexed_output;
  indexed_output.reserve(output.size());

  for (size_t i = 0; i < output.size(); ++i) {
    indexed_output.push_back({static_cast<int>(i), output[i]});
  }

  int top_k = std::min(k, static_cast<int>(indexed_output.size()));
  std::partial_sort(
      indexed_output.begin(),
      indexed_output.begin() + top_k,
      indexed_output.end(),
      [](const auto& a, const auto& b) { return a.second > b.second; });

  indexed_output.resize(top_k);
  return indexed_output;
}

/**
 * Load ImageNet class names from JSON file
 * Expected format: {"0": ["n01440764", "tench"], "1": ["n01443537", "goldfish"], ...}
 *
 * @param json_path Path to imagenet_class_index.json
 * @return Map of class_index -> class_name
 */
inline std::map<int, std::string> LoadImageNetClasses(const std::string& json_path) {
  std::map<int, std::string> class_map;
  std::ifstream file(json_path);

  if (!file.is_open()) {
    std::cerr << "[ImageNet] Failed to open: " << json_path << std::endl;
    return class_map;
  }

  std::string line;
  std::getline(file, line);  // Read entire JSON

  // Simple JSON parsing (format: {"0": ["n01440764", "tench"], ...})
  size_t pos = 0;
  while ((pos = line.find("\"", pos)) != std::string::npos) {
    size_t key_start = pos + 1;
    size_t key_end = line.find("\"", key_start);
    if (key_end == std::string::npos) break;

    std::string key_str = line.substr(key_start, key_end - key_start);

    // Skip if not a number (like "n01440764")
    if (key_str.empty() || !std::isdigit(key_str[0])) {
      pos = key_end + 1;
      continue;
    }

    int class_idx = std::stoi(key_str);

    // Find the class name (second element in array)
    size_t arr_start = line.find("[", key_end);
    if (arr_start == std::string::npos) break;

    // Skip first element (WordNet ID)
    size_t first_quote = line.find("\"", arr_start);
    size_t first_end = line.find("\"", first_quote + 1);

    // Get second element (class name)
    size_t second_quote = line.find("\"", first_end + 1);
    size_t second_end = line.find("\"", second_quote + 1);

    if (second_quote != std::string::npos && second_end != std::string::npos) {
      std::string class_name = line.substr(second_quote + 1, second_end - second_quote - 1);
      class_map[class_idx] = class_name;
    }

    pos = second_end + 1;
  }

  std::cout << "[ImageNet] Loaded " << class_map.size() << " class names" << std::endl;
  return class_map;
}

/**
 * Get class name from ImageNet class index
 *
 * @param class_map Map of class_index -> class_name
 * @param class_idx Class index
 * @return Class name, or "Unknown" if not found
 */
inline std::string GetClassName(const std::map<int, std::string>& class_map, int class_idx) {
  auto it = class_map.find(class_idx);
  if (it != class_map.end()) {
    return it->second;
  }
  return "Unknown";
}

}  // namespace utils
}  // namespace cochl_api
