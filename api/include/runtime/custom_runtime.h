// Custom runtime backend with thread pool for parallel inference.
// Provides a mock implementation for testing and demonstration.

#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "i_runtime.h"

namespace cochl_api {
namespace runtime {

/**
 * @brief Thread pool for parallel task execution
 */
class ThreadPool {
 public:
  explicit ThreadPool(size_t num_threads);
  ~ThreadPool();

  template <typename F, typename... Args>
  auto Submit(F&& f, Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      if (stop_) throw std::runtime_error("Submit on stopped ThreadPool");

      tasks_.emplace([task]() { (*task)(); });
    }
    condition_.notify_one();
    return res;
  }

  // ParallelFor: Distribute work across threads
  // Callback will be called for each range: callback(start_idx, end_idx)
  template <typename F>
  void ParallelFor(size_t start, size_t end, F&& callback) {
    if (start >= end) return;

    size_t total_work = end - start;
    size_t num_threads = workers_.size();
    size_t chunk_size = (total_work + num_threads - 1) / num_threads;

    std::vector<std::future<void>> futures;

    for (size_t t = 0; t < num_threads; ++t) {
      size_t chunk_start = start + t * chunk_size;
      size_t chunk_end = std::min(chunk_start + chunk_size, end);

      if (chunk_start >= end) break;

      auto future = Submit([callback, chunk_start, chunk_end]() {
        callback(chunk_start, chunk_end);
      });

      futures.push_back(std::move(future));
    }

    // Wait for all tasks to complete
    for (auto& future : futures) { future.get(); }
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

 private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;

  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_;
};

/**
 * @brief Custom runtime backend with thread pool
 *
 * Mock implementation for testing parallel inference execution.
 * Compatible with ResNet50 input/output dimensions.
 */
class CustomRuntime : public IRuntime {
 public:
  CustomRuntime();
  ~CustomRuntime() override;

  bool LoadModel(const char* model_path) override;
  bool RunInference(const float* input, size_t input_size,
                    float* output, size_t output_size) override;
  size_t GetInputSize() const override;
  size_t GetOutputSize() const override;
  const char* GetRuntimeType() const override;

  /**
   * @brief Set number of threads for thread pool
   * @param num_threads Number of threads to use
   */
  void SetNumThreads(size_t num_threads);

 private:
  std::unique_ptr<ThreadPool> thread_pool_;
  std::string model_path_;
  size_t input_size_;
  size_t output_size_;
  size_t num_threads_;
};

}  // namespace runtime
}  // namespace cochl_api
