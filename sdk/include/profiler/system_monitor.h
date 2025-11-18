#ifndef SDK_KERNEL_SYSTEM_MONITOR_H
#define SDK_KERNEL_SYSTEM_MONITOR_H

#include <cstddef>
#include <string>

namespace cochl {
namespace kernel {

/**
 * @brief System resource monitoring for edge devices
 *
 * Monitors memory usage, temperature, and latency across platforms:
 * - Linux (including Raspberry Pi)
 * - Windows
 * - macOS
 */
class SystemMonitor {
public:
  struct MemoryInfo {
    size_t total_bytes;       // Total system memory
    size_t available_bytes;   // Available memory
    size_t used_bytes;        // Used memory
    double usage_percent;     // Memory usage percentage
  };

  struct TemperatureInfo {
    double cpu_temp_celsius;  // CPU temperature (°C)
    bool supported;           // Temperature monitoring supported
  };

  struct LatencyInfo {
    double min_ms;            // Minimum latency
    double max_ms;            // Maximum latency
    double avg_ms;            // Average latency
    size_t sample_count;      // Number of samples
  };

  /**
   * @brief Get memory usage information
   * @return Memory info struct
   */
  static MemoryInfo GetMemoryInfo();

  /**
   * @brief Get CPU temperature
   * @return Temperature info struct
   */
  static TemperatureInfo GetTemperature();

  /**
   * @brief Record latency sample
   * @param latency_ms Latency in milliseconds
   */
  static void RecordLatency(double latency_ms);

  /**
   * @brief Get latency statistics
   * @return Latency info struct
   */
  static LatencyInfo GetLatencyInfo();

  /**
   * @brief Reset latency statistics
   */
  static void ResetLatency();

  /**
   * @brief Get formatted system status string
   * @return Status string with memory, temperature, and latency info
   */
  static std::string GetSystemStatus();

private:
  SystemMonitor() = delete;  // Static class
};

}  // namespace kernel
}  // namespace cochl

#endif  // SDK_KERNEL_SYSTEM_MONITOR_H
