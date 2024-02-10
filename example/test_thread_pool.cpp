#include <glog/logging.h>
#include <iostream>
#include "utils/thread_pool.h"

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);

    c10::ThreadPool thread_poll(c10::TaskThreadPoolBase::default_num_threads(), [](){std::cout << "init thread" << std::endl;});
    LOG(INFO) << "number of threads = " << c10::TaskThreadPoolBase::default_num_threads();
    thread_poll.run([](){std::cout << "add thread" << std::endl;});
    thread_poll.run([](){std::cout << "add thread 2" << std::endl;});
    thread_poll.run([](){std::cout << "add thread 3" << std::endl;});

    // leaked memroy, no free for now (only for testing)
    int *leaked_memory;
    thread_poll.run([&]() {
        std::cout << "allocating memory" << std::endl;
        leaked_memory = new int[100];
    });
}