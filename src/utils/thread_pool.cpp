#include <cstddef>
#include <exception>
#include <glog/logging.h>
#include <mutex>
#include <thread>

#include "utils/thread_pool.h"
#include "utils/Logging.h"


namespace c10 {

size_t TaskThreadPoolBase::default_num_threads() {
  size_t num_threads = 0;
  num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) {
    num_threads = 1;
  }
  return num_threads;
}

void ThreadPool::main_loop(size_t index) {
    LOG_INFO("thread pool start to run");
    std::unique_lock<std::mutex> lock(mutex_);
    while (is_running_) {
        condition_.wait(lock, [&](){return !tasks_.empty() || !is_running_;});
        if (!is_running_) {
            break;
        }
        {
            TaskElement tasks = std::move(tasks_.back());
            tasks_.pop_back();
            lock.unlock();

            try {
                tasks.task_body();
            } catch(const std::exception& e) {
                LOG_ERROR("Exception in thread pool task: ", e.what());
            } catch (...) {
                LOG_ERROR("Exception in thread pool task: unknown");
            }
        }
        lock.lock();
    }
}

void ThreadPool::run(std::function<void()> func) {
    TaskElement task_element(func);
    std::unique_lock<std::mutex> lock(mutex_);
    tasks_.push_back(std::move(task_element));
    condition_.notify_one();
}

ThreadPool::ThreadPool(int pool_size, const std::function<void()> init_thread)
    : threads_(pool_size), is_running_(true) {
    for (std::size_t i = 0; i < threads_.size(); ++i) {
        threads_[i] = std::thread([this, i, init_thread]() {
            if (init_thread) {
                init_thread();
            }
            this->main_loop(i);
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        is_running_ = false;
        condition_.notify_all();
    }
    for (auto& t : threads_) {
        try {
            t.join();
        } catch (const std::exception& e) {
            LOG_ERROR("thread termination error ", e.what());
        }
    } 
    LOG_INFO("thread pool ends");
}
}