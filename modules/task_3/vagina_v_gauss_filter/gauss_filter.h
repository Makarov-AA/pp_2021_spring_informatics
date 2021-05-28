// Copyright 2021 Vagina Valeria
#ifndef MODULES_TASK_3_VAGINA_V_GAUSS_FILTER_GAUSS_FILTER_H_
#define MODULES_TASK_3_VAGINA_V_GAUSS_FILTER_GAUSS_FILTER_H_

#include <cstdint>
#include <vector>

struct Pixel {
    uint8_t r, g, b;
};

bool operator==(const Pixel &a, const Pixel &b);

std::vector<Pixel> generateRandomImage(int rows, int cols);
std::vector<Pixel> gaussFilter(const std::vector<Pixel> &a,
                               int rows, int cols);
std::vector<Pixel> gaussFilterTBB(const std::vector<Pixel> &a,
    const int rows, const int cols);
std::vector<double> generateGaussianKernel(int radius);

#endif  // MODULES_TASK_3_VAGINA_V_GAUSS_FILTER_GAUSS_FILTER_H_
