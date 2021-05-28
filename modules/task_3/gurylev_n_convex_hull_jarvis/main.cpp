// Copyright 2021 Gurylev Nikita
#include <gtest/gtest.h>
#include <vector>
#include <utility>

#include "./convex_hull_jarvis.h"
#include "tbb/tick_count.h"

TEST(ConvexHull, DISABLED_test1) {
    std::vector<std::pair<int, int>> points = getRandomPoint(1550000);
    tbb::tick_count time1 = tbb::tick_count::now();
    std::vector<std::pair<int, int>> alg_seq = JarvisAlg(points);
    tbb::tick_count time2 = tbb::tick_count::now();
    std::cout << "Seq time: " <<
        static_cast<double>((time2 - time1).seconds()) << std::endl;
    time1 = tbb::tick_count::now();
    std::vector<std::pair<int, int>> alg_tbb = JarvisAlgTbb(points);
    time2 = tbb::tick_count::now();
    std::cout << "Tbb time: " <<
        static_cast<double>((time2 - time1).seconds()) << std::endl;
    ASSERT_EQ(alg_seq, alg_tbb);
}

TEST(ConvexHull, DISABLED_test2) {
    std::vector<std::pair<int, int>> points = getRandomPoint(2000000);
    tbb::tick_count time1 = tbb::tick_count::now();
    std::vector<std::pair<int, int>> alg_seq = JarvisAlg(points);
    tbb::tick_count time2 = tbb::tick_count::now();
    std::cout << "Seq time: " <<
        static_cast<double>((time2 - time1).seconds()) << std::endl;
    time1 = tbb::tick_count::now();
    std::vector<std::pair<int, int>> alg_tbb = JarvisAlgTbb(points);
    time2 = tbb::tick_count::now();
    std::cout << "Tbb time: " <<
        static_cast<double>((time2 - time1).seconds()) << std::endl;
    ASSERT_EQ(alg_seq, alg_tbb);
}

TEST(ConvexHull, test3) {
    std::vector<std::pair<int, int>> points = { std::pair<int, int>(0, 0),
        std::pair<int, int>(3, 3), std::pair<int, int>(6, 6),
        std::pair<int, int>(7, 7), std::pair<int, int>(9, 9) };
    std::vector<std::pair<int, int>> alg_seq = JarvisAlg(points);
    std::vector<std::pair<int, int>> alg_tbb = JarvisAlgTbb(points);
    ASSERT_EQ(alg_seq, alg_tbb);
}

TEST(ConvexHull, test4) {
    std::vector<std::pair<int, int>> points = { std::pair<int, int>(0, 0),
        std::pair<int, int>(34, 26), std::pair<int, int>(21, 39),
        std::pair<int, int>(0, 50), std::pair<int, int>(4, 15),
        std::pair<int, int>(36, 41), std::pair<int, int>(16, 35),
        std::pair<int, int>(8, 27), std::pair<int, int>(50, 50),
        std::pair<int, int>(50, 0) };
    std::vector<std::pair<int, int>> alg_seq = JarvisAlg(points);
    std::vector<std::pair<int, int>> alg_tbb = JarvisAlgTbb(points);
    ASSERT_EQ(alg_seq, alg_tbb);
}

TEST(ConvexHull, test5) {
    std::vector<std::pair<int, int>> points = { std::pair<int, int>(3, 8),
        std::pair<int, int>(60, 11), std::pair<int, int>(90, 40),
        std::pair<int, int>(60, 55),
        std::pair<int, int>(13, 33) };
    std::vector<std::pair<int, int>> alg_seq = JarvisAlg(points);
    std::vector<std::pair<int, int>> alg_tbb = JarvisAlgTbb(points);
    ASSERT_EQ(alg_seq, alg_tbb);
}
