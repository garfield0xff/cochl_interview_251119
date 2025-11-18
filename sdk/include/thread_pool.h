// Data-parallel thread pool implementation for performance optimization.
// Partitions work across multiple worker threads for faster computation.
// Inspired by TensorFlow Lite's threading model for neural network
// acceleration. Supports template-based callable submissions and parallel range
// operations.

#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <utility>
#include <vector>

namespace cochl {
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
}  // namespace cochl
