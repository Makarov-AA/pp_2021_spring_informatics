// Copyright 2021 Panova Olga
#ifndef MODULES_TASK_2_PANOVA_O_SHELL_B_SORT_SHELL_B_SORT_H_
#define MODULES_TASK_2_PANOVA_O_SHELL_B_SORT_SHELL_B_SORT_H_
#include <iostream>
#include <vector>
#include <random>
std::vector<int> CreateVector(int size);
std::vector<int> EvenMerge(const std::vector<int>& left, const std::vector<int>& right);
std::vector<int> OddMerge(const std::vector<int>& left, const std::vector<int>& right);
std::vector<int> BatcherMerge(const std::vector<int>& left, const std::vector<int>& right);
std::vector<int> ShellSort(const std::vector<int>& a);
std::vector<int> ParallelOddEvenShellSort(const std::vector<int>& a, int piece);
std::vector<std::vector<int>> Division(const std::vector<int>& a, int number_of_parts);
std::vector<int> ParallelOddEvenShellSort(const std::vector<int>& a, int piece);
std::vector<int> SequentialOddEvenShellSort(const std::vector<int>& a, int piece);
#endif  // MODULES_TASK_2_PANOVA_O_SHELL_B_SORT_SHELL_B_SORT_H_
