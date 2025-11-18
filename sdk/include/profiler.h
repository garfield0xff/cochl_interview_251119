#ifndef SDK_PROFILER_H
#define SDK_PROFILER_H

#include <chrono>
#include <map>
#include <string>
#include <vector>

namespace cochl_sdk {

/**
 * @brief Performance profiler for inference operations
 *
 * Tracks timing, throughput, and resource usage.
 */
class Profiler {
public:
  Profiler();
  ~Profiler();

  /**
   * @brief Start timing an operation
   * @param operation_name Name of operation to track
   */
  void StartTiming(const std::string& operation_name);

  /**
   * @brief Stop timing an operation
   * @param operation_name Name of operation to track
   */
  void StopTiming(const std::string& operation_name);

  /**
   * @brief Record inference completion
   * @param latency_ms Inference latency in milliseconds
   */
  void RecordInference(double latency_ms);

  /**
   * @brief Get statistics for all operations
   * @return Formatted statistics string
   */
  std::string GetStats() const;

  /**
   * @brief Get average inference latency
   * @return Average latency in milliseconds
   */
  double GetAverageLatency() const;

  /**
   * @brief Get throughput (inferences per second)
   * @return Throughput
   */
  double GetThroughput() const;

  /**
   * @brief Reset all statistics
   */
  void Reset();

private:
  struct TimingInfo {
    std::chrono::high_resolution_clock::time_point start_time;
    double total_time_ms;
    size_t call_count;
  };

  std::map<std::string, TimingInfo> timings_;

  // Inference statistics
  std::vector<double> inference_latencies_;
  size_t total_inferences_;
  std::chrono::high_resolution_clock::time_point first_inference_time_;
  std::chrono::high_resolution_clock::time_point last_inference_time_;
};

}  // namespace cochl_sdk

#endif  // SDK_PROFILER_H
