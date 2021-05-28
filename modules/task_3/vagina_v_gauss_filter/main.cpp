// Copyright 2021 Vagina Valeria
#include <gtest/gtest.h>
#include <tbb/tbb.h>
#include <vector>
#include <utility>
#include <numeric>
#include <iostream>
#include "../../../modules/task_3/vagina_v_gauss_filter/gauss_filter.h"

TEST(Gauss_seq, DISABLED_TimeTest) {
    int width = 1000;
    int height = 1000;

    std::vector<Pixel> img = generateRandomImage(width, height);

    tbb::tick_count t1 = tbb::tick_count::now();
    auto resultSeq = gaussFilter(img, width, height);
    tbb::tick_count t2 = tbb::tick_count::now();

    std::cout << " Sequential time " << (t2 - t1).seconds() << std::endl;

    t1 = tbb::tick_count::now();
    auto resultParallel = gaussFilterTBB(img, width, height);
    t2 = tbb::tick_count::now();

    std::cout << " Parallel time " << (t2 - t1).seconds() << std::endl;

    ASSERT_NE(resultParallel, img);
}

TEST(Gauss_seq, Small_Image) {
    int width = 10;
    int height = 10;

    std::vector<Pixel> img = generateRandomImage(width, height);

    std::vector<Pixel> res = gaussFilterTBB(img, width, height);

    ASSERT_NE(res, img);
}

TEST(Gauss_seq, Big_Image) {
    int width = 400;
    int height = 400;

    std::vector<Pixel> img = generateRandomImage(width, height);
    std::vector<Pixel> res = gaussFilterTBB(img, width, height);

    ASSERT_NE(res, img);
}

TEST(Gauss_seq, Empty_Image) {
    int width = 0;
    int height = 0;

    std::vector<Pixel> img = generateRandomImage(width, height);

    ASSERT_EQ(img, gaussFilterTBB(img, width, height));
}

TEST(Gauss_seq, Kernel_Size) {
    int width = 2;
    int height = 50;

    std::vector<Pixel> img = generateRandomImage(width, height);

    ASSERT_NE(img, gaussFilterTBB(img, width, height));
}

TEST(Gauss_seq, Kernel_Size_1) {
    int width = 50;
    int height = 2;

    std::vector<Pixel> img = generateRandomImage(width, height);

    ASSERT_NE(img, gaussFilterTBB(img, width, height));
}

TEST(Gauss_seq, Incorrect_Image_Height) {
    int width = 10;
    int height = 10;

    std::vector<Pixel> img = generateRandomImage(width, height);

    ASSERT_ANY_THROW(gaussFilterTBB(img, width, height - 1));
}

TEST(Gauss_seq, Incorrect_Image_Width) {
    int width = 10;
    int height = 10;

    std::vector<Pixel> img = generateRandomImage(width, height);

    ASSERT_ANY_THROW(gaussFilterTBB(img, width + 1, height));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
