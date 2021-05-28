// Copyright 2021 Streltsova Yana
#ifndef MODULES_TASK_3_STRELTSOVA_Y_SHELL_SORT_SHELL_SORT_H_
#define MODULES_TASK_3_STRELTSOVA_Y_SHELL_SORT_SHELL_SORT_H_

#include <tbb/tbb.h>
#include <utility>
#include <vector>

std::vector<int> getRandomVectorInt(int  sz);

std::vector<double> getRandomVectorDouble(int  sz);

template< typename T >
std::vector<T> shell_sort_sequential(const std::vector<T>& vec) {
    std::vector<T> result(vec);
    int d = 4;
    while (d > 0) {
        for (size_t i = 0; i < result.size(); i++) {
            for (size_t j = i + d; j < result.size(); j += d) {
                if (result[i] > result[j])
                    std::swap(result[i], result[j]);
            }
        }
        d /= 2;
    }
    return result;
}

template< typename T >
std::vector<T> shell_sort_parallel(const std::vector<T>& vec) {
    std::vector<T> result(vec);
    if (vec.size() < 1000)
        return shell_sort_sequential(vec);
    int d = 4;
    while (d > 0) {
        tbb::task_scheduler_init init(d);
        int tid = tbb::task_arena::current_thread_index();
            for (size_t i = tid; i < result.size(); i += d)
                for (size_t j = i + d; j < result.size(); j += d) {
                    if (result[i] > result[j])
                        std::swap(result[i], result[j]);
                }
        init.terminate();
        d /= 2;
    }
    return result;
}

#endif  // MODULES_TASK_3_STRELTSOVA_Y_SHELL_SORT_SHELL_SORT_H_
