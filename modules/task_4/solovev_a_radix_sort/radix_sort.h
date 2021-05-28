// Copyright 2021 Solovev Aleksandr
#ifndef MODULES_TASK_4_SOLOVEV_A_RADIX_SORT_RADIX_SORT_H_
#define MODULES_TASK_4_SOLOVEV_A_RADIX_SORT_RADIX_SORT_H_

#include <time.h>
#include <vector>
#include <random>

#include "../../../3rdparty/unapproved/unapproved.h"

int generateRandomArray(std::vector<int> *buffer, int min, int max);
int SortingCheck(std::vector<int> *buffer);
int RadixSort(std::vector<int> *buffer);
int ParallelSortingSTD(std::vector<int> *buffer, int thread_count = std::thread::hardware_concurrency());
#endif  // MODULES_TASK_4_SOLOVEV_A_RADIX_SORT_RADIX_SORT_H_
