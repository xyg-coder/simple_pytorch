#include <future>
#include <glog/logging.h>
#include <iostream>
#include "utils/CallOnce.h"
#include "utils/Logging.h"
#include "utils/thread_pool.h"

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);

    c10::ThreadPool thread_poll(c10::TaskThreadPoolBase::default_num_threads(), [](){std::cout << "init thread" << std::endl;});
    LOG_INFO("number of threads = ", c10::TaskThreadPoolBase::default_num_threads());
    thread_poll.run([](){std::cout << "add thread" << std::endl;});
    thread_poll.run([](){std::cout << "add thread 2" << std::endl;});
    thread_poll.run([](){std::cout << "add thread 3" << std::endl;});

    c10::OnceFlag once_flag;

    for (int i = 0; i < 10; ++i) {
        thread_poll.run([&]{
            c10::callOnce(once_flag, []{
                std::cout << "should only call once" << std::endl;
            });
        });
        thread_poll.run([&]{
            std::cout << "should call ten times" << std::endl;
        });
    }

    int *allocated_mem;
    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    thread_poll.run([&]() {
        std::cout << "allocating memory" << std::endl;
        allocated_mem = new int[100];
        allocated_mem[0] = 0;
        promise.set_value();
    });
    future.wait();
    std::cout << "allocated_mem: " << allocated_mem[0] << std::endl;
    delete []allocated_mem;
}
