// Copyright 2021 Paranicheva Alyona
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <ctime>
#include "./labeling.h"

TEST(CLabeling_TBB, 7x10) {
    int rows = 15;
    int cols = 17;
    std::vector<int> img = getRandomMatrix(rows, cols);
    std::vector<int> expres = CLabeling(img, rows, cols);
    std::vector<int> res = CLabelingTBB(img, rows, cols);
    ASSERT_EQ(expres, res);
}

TEST(CLabeling_TBB, 15x17) {
    int rows = 15;
    int cols = 17;
    std::vector<int> img = getRandomMatrix(rows, cols);
    std::vector<int> expres = CLabeling(img, rows, cols);
    std::vector<int> res = CLabelingTBB(img, rows, cols);
    ASSERT_EQ(expres, res);
}

TEST(CLabeling_TBB, 21x8) {
    int rows = 21;
    int cols = 8;
    std::vector<int> img = getRandomMatrix(rows, cols);
    std::vector<int> expres = CLabeling(img, rows, cols);
    std::vector<int> res = CLabelingTBB(img, rows, cols);
    ASSERT_EQ(expres, res);
}

TEST(CLabeling_TBB, 23x15) {
    int rows = 21;
    int cols = 8;
    std::vector<int> img = getRandomMatrix(rows, cols);
    std::vector<int> expres = CLabeling(img, rows, cols);
    std::vector<int> res = CLabelingTBB(img, rows, cols);
    ASSERT_EQ(expres, res);
}

TEST(Clabeling_Matrix_gen, Invalid_size) {
    int rows = -5;
    int cols = 7;
    ASSERT_ANY_THROW(getRandomMatrix(rows, cols));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
