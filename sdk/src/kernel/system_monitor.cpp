#include "kernel/system_monitor.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>

#ifdef __linux__
#include <sys/sysinfo.h>
#include <unistd.h>
#elif _WIN32
#include <windows.h>
#include <psapi.h>
#elif __APPLE__
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

namespace cochl {
namespace kernel {

// Static storage for latency tracking
static std::vector<double> latency_samples;
static constexpr size_t MAX_SAMPLES = 1000;

SystemMonitor::MemoryInfo SystemMonitor::GetMemoryInfo() {
  MemoryInfo info{};

#ifdef __linux__
  struct sysinfo si;
  if (sysinfo(&si) == 0) {
    info.total_bytes     = si.totalram * si.mem_unit;
    info.available_bytes = si.freeram * si.mem_unit;
    info.used_bytes      = (si.totalram - si.freeram) * si.mem_unit;
    info.usage_percent   = (info.used_bytes * 100.0) / info.total_bytes;
  }

#elif _WIN32
  MEMORYSTATUSEX memInfo;
  memInfo.dwLength = sizeof(MEMORYSTATUSEX);
  if (GlobalMemoryStatusEx(&memInfo)) {
    info.total_bytes     = memInfo.ullTotalPhys;
    info.available_bytes = memInfo.ullAvailPhys;
    info.used_bytes      = info.total_bytes - info.available_bytes;
    info.usage_percent   = memInfo.dwMemoryLoad;
  }

#elif __APPLE__
  int mib[2] = {CTL_HW, HW_MEMSIZE};
  size_t length = sizeof(info.total_bytes);
  sysctl(mib, 2, &info.total_bytes, &length, nullptr, 0);

  vm_statistics64_data_t vm_stats;
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm_stats,
                        &count) == KERN_SUCCESS) {
    info.available_bytes = (vm_stats.free_count + vm_stats.inactive_count) *
                           vm_page_size;
    info.used_bytes    = info.total_bytes - info.available_bytes;
    info.usage_percent = (info.used_bytes * 100.0) / info.total_bytes;
  }
#endif

  return info;
}

SystemMonitor::TemperatureInfo SystemMonitor::GetTemperature() {
  TemperatureInfo info{};
  info.supported      = false;
  info.cpu_temp_celsius = 0.0;

#ifdef __linux__
  // Try to read from thermal zone (common on Raspberry Pi and Linux)
  std::ifstream temp_file("/sys/class/thermal/thermal_zone0/temp");
  if (temp_file.is_open()) {
    int temp_millidegrees;
    temp_file >> temp_millidegrees;
    info.cpu_temp_celsius = temp_millidegrees / 1000.0;
    info.supported      = true;
  }

  // Fallback: Try Raspberry Pi specific location
  if (!info.supported) {
    std::ifstream rpi_temp("/sys/class/hwmon/hwmon0/temp1_input");
    if (rpi_temp.is_open()) {
      int temp_millidegrees;
      rpi_temp >> temp_millidegrees;
      info.cpu_temp_celsius = temp_millidegrees / 1000.0;
      info.supported      = true;
    }
  }

#elif __APPLE__
  // macOS temperature monitoring requires IOKit (complex)
  // Leave as unsupported for now
  info.supported = false;

#elif _WIN32
  // Windows temperature monitoring requires WMI or specific drivers
  // Leave as unsupported for now
  info.supported = false;
#endif

  return info;
}

void SystemMonitor::RecordLatency(double latency_ms) {
  latency_samples.push_back(latency_ms);

  // Keep only last MAX_SAMPLES
  if (latency_samples.size() > MAX_SAMPLES) {
    latency_samples.erase(latency_samples.begin());
  }
}

SystemMonitor::LatencyInfo SystemMonitor::GetLatencyInfo() {
  LatencyInfo info{};

  if (latency_samples.empty()) {
    info.min_ms       = 0.0;
    info.max_ms       = 0.0;
    info.avg_ms       = 0.0;
    info.sample_count = 0;
    return info;
  }

  info.sample_count = latency_samples.size();
  info.min_ms = *std::min_element(latency_samples.begin(), latency_samples.end());
  info.max_ms = *std::max_element(latency_samples.begin(), latency_samples.end());

  double sum = 0.0;
  for (double lat : latency_samples) {
    sum += lat;
  }
  info.avg_ms = sum / latency_samples.size();

  return info;
}

void SystemMonitor::ResetLatency() { latency_samples.clear(); }

std::string SystemMonitor::GetSystemStatus() {
  std::ostringstream oss;

  // Memory information
  auto mem = GetMemoryInfo();
  oss << "Memory: " << (mem.used_bytes / (1024 * 1024)) << " MB / "
      << (mem.total_bytes / (1024 * 1024)) << " MB (" << mem.usage_percent << "%)\n";

  // Temperature information
  auto temp = GetTemperature();
  if (temp.supported) {
    oss << "CPU Temperature: " << temp.cpu_temp_celsius << " °C\n";
  } else {
    oss << "CPU Temperature: Not supported on this platform\n";
  }

  // Latency information
  auto latency = GetLatencyInfo();
  if (latency.sample_count > 0) {
    oss << "Latency: min=" << latency.min_ms << " ms, max=" << latency.max_ms
        << " ms, avg=" << latency.avg_ms << " ms (" << latency.sample_count
        << " samples)\n";
  }

  return oss.str();
}

}  // namespace kernel
}  // namespace cochl
