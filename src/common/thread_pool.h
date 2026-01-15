#pragma once

/// @file thread_pool.h
/// @brief PyFlare thread pool for parallel task execution

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <vector>

namespace pyflare {

/// @brief A simple thread pool for executing tasks in parallel
class ThreadPool {
public:
    /// @brief Create a thread pool with the specified number of threads
    /// @param num_threads Number of worker threads (default: hardware concurrency)
    explicit ThreadPool(size_t num_threads = 0);

    /// @brief Destructor - waits for all tasks to complete
    ~ThreadPool();

    // Non-copyable and non-movable
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    /// @brief Submit a task for execution
    /// @param f Callable to execute
    /// @param args Arguments to pass to the callable
    /// @return Future containing the result
    template <typename F, typename... Args>
    auto Submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>;

    /// @brief Submit a task without waiting for result
    /// @param f Callable to execute
    /// @param args Arguments to pass to the callable
    template <typename F, typename... Args>
    void Execute(F&& f, Args&&... args);

    /// @brief Get the number of worker threads
    size_t Size() const { return workers_.size(); }

    /// @brief Get the number of pending tasks
    size_t PendingTasks() const;

    /// @brief Wait for all submitted tasks to complete
    void Wait();

    /// @brief Check if the pool is stopped
    bool IsStopped() const { return stop_.load(std::memory_order_acquire); }

private:
    void WorkerLoop();

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::condition_variable completion_condition_;

    std::atomic<bool> stop_{false};
    std::atomic<size_t> active_tasks_{0};
};

// Template implementations

template <typename F, typename... Args>
auto ThreadPool::Submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
    using return_type = std::invoke_result_t<F, Args...>;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> result = task->get_future();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_.load(std::memory_order_acquire)) {
            throw std::runtime_error("Cannot submit task to stopped thread pool");
        }
        tasks_.emplace([task]() { (*task)(); });
    }

    condition_.notify_one();
    return result;
}

template <typename F, typename... Args>
void ThreadPool::Execute(F&& f, Args&&... args) {
    auto task = std::bind(std::forward<F>(f), std::forward<Args>(args)...);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_.load(std::memory_order_acquire)) {
            throw std::runtime_error("Cannot execute task on stopped thread pool");
        }
        tasks_.emplace(std::move(task));
    }

    condition_.notify_one();
}

/// @brief Get the global thread pool instance
/// @param num_threads Number of threads (only used on first call)
/// @return Reference to the global thread pool
ThreadPool& GlobalThreadPool(size_t num_threads = 0);

}  // namespace pyflare
