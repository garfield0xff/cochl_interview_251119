#include "thread_pool.h"

#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

// TODO: 구현 필요
//
// ThreadPool::ThreadPool(size_t num_threads) 생성자
//   - num_threads개 워커 스레드 생성
//   - 각 워커는 다음을 반복:
//     1. 큐(tasks_)에서 작업 가져오기
//     2. 조건변수(condition_)로 새 작업 대기
//     3. 작업 실행
//     4. stop_ && tasks_.empty()이면 종료
//
// ThreadPool::~ThreadPool() 소멸자
//   - stop_ = true 설정
//   - 모든 워커에게 condition_variable 신호
//   - 모든 워커 스레드 join()
//
// 필요 멤버 변수:
//   - std::vector<std::thread> workers_
//   - std::queue<std::function<void()>> tasks_
//   - std::mutex queue_mutex_
//   - std::condition_variable condition_
//   - bool stop_
