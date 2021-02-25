//
// Created by syys on 2021/2/25.
//

#ifndef EXAMPLE_APP_THREAD_POOL_H
#define EXAMPLE_APP_THREAD_POOL_H

#include <vector>
#include <queue>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <future>
#include <functional>
#include "core/global.h"

class ThreadPool
{
public:
    using Task = std::function<void ()>;

    explicit ThreadPool(unsigned int n_thread = 4);

    ~ThreadPool();

    int get_idle_num();

    template<class F, class... Args>
    auto commit(F &&f, Args &&... args)->std::future<decltype(f(args...))>
    {
        // 提交任务，返回 std::future
        if (!this->run) throw std::runtime_error("commit error : ThreadPool is stopped.");
        // using return_type = typename std::result_of<F(Args...)>::type;
        using return_type = decltype(f(args...));
        // packaged_task package the bind function and future
        auto task_ptr = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        {
            std::lock_guard<std::mutex> lock_temp(this->look);
            this->tasks.emplace([task_ptr]() { (*task_ptr)(); });
        }

        this->cv.notify_one();
        return task_ptr->get_future();
    }

private:
    // 线程池
    std::vector<std::thread> pool;
    // 任务队列
    std::queue<Task> tasks;
    // 同步
    std::mutex look;
    std::condition_variable cv;
    // 状态
    std::atomic_bool run;
    std::atomic_uint idle_num;
};

#endif //EXAMPLE_APP_THREAD_POOL_H
