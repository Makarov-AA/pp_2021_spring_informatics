// Copyright 2021 Lyutova Tanya
#include <gtest/gtest.h>
#include <tbb/tick_count.h>
#include "../../modules/task_3/lyutova_t_sobel/lyutova_t_sobel.h"

TEST(TBB_Sobel, Performance) {
    Image image = createRandomImage(5000, 5000);
    tbb::tick_count t1, t2;
    t1 = tbb::tick_count::now();
    Image seq = sobel(image);
    t2 = tbb::tick_count::now();
    std::cout << "seq " << (t2 - t1).seconds() << std::endl;
    t1 = tbb::tick_count::now();
    Image par = sobelParallel(image);
    t2 = tbb::tick_count::now();
    std::cout << "par " << (t2 - t1).seconds() << std::endl;
    ASSERT_EQ(seq, par);
}

TEST(TBB_Sobel, RandomImage) {
    Image image = createRandomImage(100, 100);
    ASSERT_EQ(image.cols, 100);
    ASSERT_EQ(image.rows, 100);
}

TEST(TBB_Sobel, SameAsSequential) {
  Image image = createRandomImage(100, 100);
  Image seq = sobel(image);
  Image par = sobelParallel(image);
  ASSERT_EQ(seq, par);
}

TEST(TBB_Sobel, DifferentImages) {
    Image image = createRandomImage(100, 100);
    Image res = sobelParallel(image);
    ASSERT_NE(image.pixels, res.pixels);
}

TEST(TBB_Sobel, SameResult) {
    Image image(4, 4);
    Image seq = sobel(image);
    Image par = sobelParallel(image);
    ASSERT_EQ(seq, par);
}

TEST(TBB_Sobel, Different) {
    Image image = createRandomImage(100, 100);
    Image image2(100, 100);
    ASSERT_NE(image.pixels, image2.pixels);
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
