//
// Created by syys on 2021/2/25.
//

#include "thread_pool.h"


ThreadPool::~ThreadPool() {
    this->run = false;
    // wake 所有正在wait的线程
    this->cv.notify_all();
    for (std::thread & t : this->pool) t.join();
}


int ThreadPool::get_idle_num() {
    return idle_num;
}

ThreadPool::ThreadPool(unsigned int n_thread): run(true), idle_num(n_thread)
{
    if (n_thread < 1) {
        throw StringError("thread number less than 1 is illegal!");
    }
    // 创建线程
    for (unsigned int i = 0; i < n_thread; i++)
    {
        // 匿名函数作为创建线程的参数
        pool.emplace_back([this]() {
            while(this->run)
            {
                Task task;
                {
                    std::unique_lock<std::mutex> lock_temp(this->look);
                    // wait 之前判断匿名函数的返回值，为true时不wait，为false时wait
                    // wait 过程中（锁已经释放）需要 notify 才会退出 wait，重新加锁
                    this->cv.wait(lock_temp, [this]() { return !this->tasks.empty() || !this->run; });
                    // 禁用并且任务队列为空时，退出线程
                    if (!this->run && this->tasks.empty()) return;

                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }
                // 执行
                this->idle_num--;
                task();
                this->idle_num++;
            }
        });
    }
}

