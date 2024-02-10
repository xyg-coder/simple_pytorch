#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace c10 {
class TaskThreadPoolBase {
public:
    virtual void run(std::function<void()> func) = 0;
    virtual ~TaskThreadPoolBase() noexcept = default;
    static size_t default_num_threads();
};

struct TaskElement {
    const std::function<void()> task_body;
    explicit TaskElement(std::function<void()> f): task_body(std::move(f)) {}
};

class ThreadPool : public TaskThreadPoolBase {
public:
    ThreadPool() = delete;
    explicit ThreadPool(int pool_size, const std::function<void()> init_thread);
    void run(std::function<void()> func) override;
    ~ThreadPool() override;
private:
    // entry point for pool threads
    // each thread will run this function
    void main_loop(size_t index);
    std::vector<TaskElement> tasks_;
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable condition_;
    bool is_running_;
};
}