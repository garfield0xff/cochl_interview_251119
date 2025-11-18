#ifndef SDK_KERNEL_SYSTEM_MONITOR_H
#define SDK_KERNEL_SYSTEM_MONITOR_H

#include <cstddef>
#include <string>

namespace cochl {
namespace kernel {

/**
 * @brief System resource monitoring for edge devices
 */
class SystemMonitor {
public:
  struct MemoryInfo {
    size_t total_bytes;       // Total system memory
    size_t available_bytes;   // Available memory
    size_t used_bytes;        // Used memory
    double usage_percent;     // Memory usage percentage
  };

  struct LatencyInfo {
    double min_ms;            // Minimum latency
    double max_ms;            // Maximum latency
    double avg_ms;            // Average latency
    size_t sample_count;      // Number of samples
  };

  static MemoryInfo GetMemoryInfo();

  static void RecordLatency(double latency_ms);

  static LatencyInfo GetLatencyInfo();

  static void ResetLatency();

  static std::string GetSystemStatus();

private:
  SystemMonitor() = delete;  
};

}  // namespace kernel
}  // namespace cochl

#endif  // SDK_KERNEL_SYSTEM_MONITOR_H
