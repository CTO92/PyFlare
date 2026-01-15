/// @file thread_pool_test.cpp
/// @brief Tests for PyFlare thread pool

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <vector>

#include "common/thread_pool.h"

namespace pyflare {
namespace {

TEST(ThreadPoolTest, BasicExecution) {
    ThreadPool pool(2);

    auto future = pool.Submit([]() { return 42; });

    EXPECT_EQ(future.get(), 42);
}

TEST(ThreadPoolTest, MultipleSubmissions) {
    ThreadPool pool(4);

    std::vector<std::future<int>> futures;
    for (int i = 0; i < 10; ++i) {
        futures.push_back(pool.Submit([i]() { return i * 2; }));
    }

    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(futures[i].get(), i * 2);
    }
}

TEST(ThreadPoolTest, ConcurrentExecution) {
    ThreadPool pool(4);
    std::atomic<int> counter{0};

    std::vector<std::future<void>> futures;
    for (int i = 0; i < 100; ++i) {
        futures.push_back(pool.Submit([&counter]() {
            counter.fetch_add(1, std::memory_order_relaxed);
        }));
    }

    for (auto& f : futures) {
        f.get();
    }

    EXPECT_EQ(counter.load(), 100);
}

TEST(ThreadPoolTest, Wait) {
    ThreadPool pool(2);
    std::atomic<int> counter{0};

    for (int i = 0; i < 10; ++i) {
        pool.Execute([&counter]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            counter.fetch_add(1);
        });
    }

    pool.Wait();

    EXPECT_EQ(counter.load(), 10);
}

TEST(ThreadPoolTest, PendingTasks) {
    ThreadPool pool(1);

    // Submit tasks that take some time
    for (int i = 0; i < 5; ++i) {
        pool.Execute([]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        });
    }

    // Should have pending tasks
    EXPECT_GT(pool.PendingTasks(), 0u);

    pool.Wait();

    // All done
    EXPECT_EQ(pool.PendingTasks(), 0u);
}

TEST(ThreadPoolTest, ExceptionHandling) {
    ThreadPool pool(2);

    auto future = pool.Submit([]() -> int {
        throw std::runtime_error("Test exception");
    });

    EXPECT_THROW(future.get(), std::runtime_error);
}

TEST(ThreadPoolTest, Size) {
    ThreadPool pool(8);
    EXPECT_EQ(pool.Size(), 8u);
}

TEST(ThreadPoolTest, DefaultSize) {
    ThreadPool pool;
    EXPECT_GT(pool.Size(), 0u);
}

TEST(ThreadPoolTest, GlobalThreadPool) {
    auto& pool = GlobalThreadPool(4);

    auto future = pool.Submit([]() { return 123; });
    EXPECT_EQ(future.get(), 123);
}

}  // namespace
}  // namespace pyflare
