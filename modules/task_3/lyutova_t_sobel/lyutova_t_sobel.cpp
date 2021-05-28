// Copyright 2021 Lyutova Tanya
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <random>
#include <cmath>
#include <vector>
#include "../../modules/task_3/lyutova_t_sobel/lyutova_t_sobel.h"

inline int clamp(int value, int min, int max) {
    if (value > max)
        return max;
    if (value < min)
        return min;
    return value;
}

Image sobel(const Image &image) {
    static const std::vector<int> matrixX = { -1, 2, -1, 0, 0, 0, 1, 2, 1 };
    static const std::vector<int> matrixY = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    static const int matrixSize = 3;
    static const int matrixRadius = 1;
    Image filteredImage(image.rows, image.cols);
    for (int x = 0; x < image.rows; x++)
        for (int y = 0; y < image.cols; y++) {
            int valX = 0, valY = 0;
            for (int i = -matrixRadius; i <= matrixRadius; i++)
                for (int j = -matrixRadius; j <= matrixRadius; j++) {
                    int index = (i + matrixRadius) * matrixSize + j +
                        matrixRadius;
                    int nearX = clamp(x + j, 0, image.rows - 1);
                    int nearY = clamp(y + i, 0, image.cols - 1);
                    valX += image(nearX, nearY) * matrixX[index];
                    valY += image(nearX, nearY) * matrixY[index];
                }
            filteredImage(x, y) =
                clamp(static_cast<int>(std::sqrt(valX * valX +
                    valY * valY)), 0, 255);
        }
    return filteredImage;
}

Image sobelParallel(const Image &image) {
    static const std::vector<int> matrixX = { -1, 2, -1, 0, 0, 0, 1, 2, 1 };
    static const std::vector<int> matrixY = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    static const int matrixSize = 3;
    static const int matrixRadius = 1;
    Image filteredImage(image.rows, image.cols);
    tbb::parallel_for(tbb::blocked_range<int>(0, image.rows),
        [&](const tbb::blocked_range<int>& r) {
        for (int x = r.begin(); x < r.end(); x++)
            for (int y = 0; y < image.cols; y++) {
                int valX = 0, valY = 0;
                for (int i = -matrixRadius; i <= matrixRadius; i++)
                    for (int j = -matrixRadius; j <= matrixRadius; j++) {
                        int index = (i + matrixRadius) * matrixSize + j +
                            matrixRadius;
                        int nearX = clamp(x + j, 0, image.rows - 1);
                        int nearY = clamp(y + i, 0, image.cols - 1);
                        valX += image(nearX, nearY) * matrixX[index];
                        valY += image(nearX, nearY) * matrixY[index];
                    }
                filteredImage(x, y) = clamp(
                    static_cast<int>(std::sqrt(valX * valX +
                        valY * valY)), 0, 255);
            }
        });
    return filteredImage;
}

Image createRandomImage(int rows, int cols) {
    Image image(rows, cols);
    std::random_device rd;
    std::mt19937 generator(rd());
    for (size_t i = 0; i < image.pixels.size(); i++)
        image.pixels[i] = std::abs(static_cast<int>(generator())) % 256;
    return image;
}
