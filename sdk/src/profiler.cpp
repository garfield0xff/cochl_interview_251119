#include "profiler.h"

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace cochl_sdk {

Profiler::Profiler() : total_inferences_(0) {}

Profiler::~Profiler() = default;

void Profiler::StartTiming(const std::string& operation_name) {
  auto& timing = timings_[operation_name];
  timing.start_time = std::chrono::high_resolution_clock::now();
}

void Profiler::StopTiming(const std::string& operation_name) {
  auto end_time = std::chrono::high_resolution_clock::now();

  auto it = timings_.find(operation_name);
  if (it == timings_.end() || it->second.start_time.time_since_epoch().count() == 0) {
    return;  // No matching start time
  }

  auto& timing = it->second;
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - timing.start_time);

  timing.total_time_ms += duration.count() / 1000.0;
  timing.call_count++;
  timing.start_time = std::chrono::high_resolution_clock::time_point{};  // Reset
}

void Profiler::RecordInference(double latency_ms) {
  inference_latencies_.push_back(latency_ms);
  total_inferences_++;

  if (total_inferences_ == 1) {
    first_inference_time_ = std::chrono::high_resolution_clock::now();
  }
  last_inference_time_ = std::chrono::high_resolution_clock::now();

  // Keep only last 100 latencies to avoid unbounded growth
  if (inference_latencies_.size() > 100) {
    inference_latencies_.erase(inference_latencies_.begin());
  }
}

std::string Profiler::GetStats() const {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);

  oss << "\n=== Profiling Statistics ===\n";

  // Operation timings
  if (!timings_.empty()) {
    oss << "\nOperation Timings:\n";
    for (const auto& pair : timings_) {
      const auto& name = pair.first;
      const auto& timing = pair.second;

      if (timing.call_count > 0) {
        double avg = timing.total_time_ms / timing.call_count;
        oss << "  " << name << ": " << avg << " ms (avg), " << timing.call_count
            << " calls\n";
      }
    }
  }

  // Inference statistics
  if (total_inferences_ > 0) {
    oss << "\nInference Statistics:\n";
    oss << "  Total inferences: " << total_inferences_ << "\n";
    oss << "  Average latency: " << GetAverageLatency() << " ms\n";
    oss << "  Throughput: " << GetThroughput() << " inferences/sec\n";
  }

  return oss.str();
}

double Profiler::GetAverageLatency() const {
  if (inference_latencies_.empty()) {
    return 0.0;
  }

  double sum = std::accumulate(inference_latencies_.begin(), inference_latencies_.end(), 0.0);
  return sum / inference_latencies_.size();
}

double Profiler::GetThroughput() const {
  if (total_inferences_ < 2) {
    return 0.0;
  }

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      last_inference_time_ - first_inference_time_);

  if (duration.count() == 0) {
    return 0.0;
  }

  return (total_inferences_ * 1000.0) / duration.count();
}

void Profiler::Reset() {
  timings_.clear();
  inference_latencies_.clear();
  total_inferences_ = 0;
}

}  // namespace cochl_sdk
