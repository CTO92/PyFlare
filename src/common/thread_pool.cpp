#include "thread_pool.h"

namespace pyflare {

ThreadPool::ThreadPool(size_t num_threads) {
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) {
            num_threads = 4;  // Fallback
        }
    }

    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back(&ThreadPool::WorkerLoop, this);
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_.store(true, std::memory_order_release);
    }

    condition_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::WorkerLoop() {
    while (true) {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(mutex_);

            condition_.wait(lock, [this] {
                return stop_.load(std::memory_order_acquire) || !tasks_.empty();
            });

            if (stop_.load(std::memory_order_acquire) && tasks_.empty()) {
                return;
            }

            task = std::move(tasks_.front());
            tasks_.pop();
            active_tasks_.fetch_add(1, std::memory_order_relaxed);
        }

        task();

        active_tasks_.fetch_sub(1, std::memory_order_relaxed);
        completion_condition_.notify_all();
    }
}

size_t ThreadPool::PendingTasks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tasks_.size() + active_tasks_.load(std::memory_order_relaxed);
}

void ThreadPool::Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    completion_condition_.wait(lock, [this] {
        return tasks_.empty() && active_tasks_.load(std::memory_order_relaxed) == 0;
    });
}

ThreadPool& GlobalThreadPool(size_t num_threads) {
    static ThreadPool pool(num_threads);
    return pool;
}

}  // namespace pyflare
