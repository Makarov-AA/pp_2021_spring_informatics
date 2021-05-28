// Copyright 2021 Solovev Aleksandr
#ifndef MODULES_TASK_3_SOLOVEV_A_RADIX_SORT_RADIX_SORT_H_
#define MODULES_TASK_3_SOLOVEV_A_RADIX_SORT_RADIX_SORT_H_

#include <time.h>
#include <vector>
#include <random>

int generateRandomArray(std::vector<int> *buffer, int min, int max);
int SortingCheck(std::vector<int> *buffer);
int RadixSort(std::vector<int> *buffer);
int ParallelSortingTBB(std::vector<int> *buffer);

#endif  // MODULES_TASK_3_SOLOVEV_A_RADIX_SORT_RADIX_SORT_H_
