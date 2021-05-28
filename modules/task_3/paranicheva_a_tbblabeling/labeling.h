// Copyright 2021 Paranicheva Alyona
#ifndef MODULES_TASK_3_PARANICHEVA_A_TBBLABELING_LABELING_H_
#define MODULES_TASK_3_PARANICHEVA_A_TBBLABELING_LABELING_H_

#include <tbb/tbb.h>
#include <vector>
#include <utility>
#include "tbb/parallel_for.h"

std::vector<int> getRandomMatrix(int rows, int cols);

std::pair<std::vector<int>, std::vector<int> > firstMark(std::vector<int>, int, int);
std::vector<int> secondMark(std::vector<int>, int, int, std::vector<int>);
std::vector<int> CLabeling(const std::vector<int>&, int, int);
std::vector<int> Transform(const std::vector<int>& array, int rows, int cols);

std::pair<std::vector<int>, std::vector<int> > firstMarkTBB(std::vector<int> arr, int rows, int cols);
std::vector<int> CLabelingTBB(const std::vector<int>&, int, int);

class FirstMarkTBB {
    std::vector<int>* arr;
    std::vector<int>* sets;
    std::vector<int> strbeg;
    std::vector<int> kolvo;
    int cols;

 public:
    FirstMarkTBB(std::vector<int>* _arr, std::vector<int>* _sets, const std::vector<int>& _strbeg,
        const std::vector<int>& _kolvo, int _cols) :
        arr(_arr), sets(_sets), strbeg(_strbeg), kolvo(_kolvo), cols(_cols) {}

    void operator() (const tbb::blocked_range<int>& range) const {
        for (int i = strbeg[range.begin()]; i < strbeg[range.begin()] + kolvo[range.begin()]; i++) {
            for (int j = 1; j < cols - 1; j++) {
                if ((*arr)[i * cols + j] == 0)
                    continue;
                if (((*arr)[i * cols + j - 1] == 0) && (((*arr)[(i - 1) * cols + j] == 0) ||
                                                        (i == strbeg[range.begin()]))) {
                    (*arr)[i * cols + j] = i * cols + j + 1;
                    continue;
                }
                if (((*arr)[i * cols + j - 1] != 0) && (((*arr)[(i - 1) * cols + j] == 0) ||
                                                        (i == strbeg[range.begin()]))) {
                    (*arr)[i * cols + j] = (*arr)[i * cols + j - 1];
                    continue;
                }
                if (((*arr)[i * cols + j - 1] == 0) && ((*arr)[(i - 1) * cols + j] != 0)) {
                    (*arr)[i * cols + j] = (*arr)[(i - 1) * cols + j];
                    continue;
                }
                if (((*arr)[i * cols + j - 1] != 0) && ((*arr)[(i - 1) * cols + j] != 0)) {
                    int max, min;
                    if ((*arr)[i * cols + j - 1] < (*arr)[(i - 1) * cols + j]) {
                        max = (*arr)[(i - 1) * cols + j];
                        min = (*arr)[i * cols + j - 1];
                    } else {
                        max = (*arr)[i * cols + j - 1];
                        min = (*arr)[(i - 1) * cols + j];
                    }
                    while ((*sets)[max] != max) {
                        max = (*sets)[max];
                    }
                    while ((*sets)[min] != min)
                        min = (*sets)[min];
                    if (min != max)
                        (*sets)[max] = min;
                    (*arr)[i * cols + j] = min;
                }
            }
        }
    }
};

#endif  // MODULES_TASK_3_PARANICHEVA_A_TBBLABELING_LABELING_H_
