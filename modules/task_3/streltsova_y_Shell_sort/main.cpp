// Copyright 2021 Streltsova Yana

#include <gtest/gtest.h>
#include <tbb/tick_count.h>

#include <algorithm>

#include "./Shell_sort.h"


TEST(Shell_sort_tbb, Test_zero_size) {
    const int size_vector = 0;
    std::vector<int> a = getRandomVectorInt(size_vector);
    std::vector<int> result;
    std::vector<int> parallel_a = shell_sort_parallel(a);
    ASSERT_EQ(result, parallel_a);
}

TEST(Shell_sort_tbb, Test_one_element) {
    const int size_vector = 1;
    std::vector<int> a = getRandomVectorInt(size_vector);
    std::vector<int> result(a);
    std::vector<int> parallel_a = shell_sort_parallel(a);
    ASSERT_EQ(result, parallel_a);
}

TEST(Shell_sort_tbb, Test_sorted_vector) {
    const int size_vector = 100;
    std::vector<int> a = getRandomVectorInt(size_vector);
    std::sort(a.begin(), a.end());
    std::vector<int> result(a);
    std::vector<int> parallel_a = shell_sort_parallel(a);
    ASSERT_EQ(result, parallel_a);
}

TEST(Shell_sort_tbb, Test_normal_size) {
    const int size_vector = 100;
    std::vector<int> a = getRandomVectorInt(size_vector);
    tbb::tick_count t1, t2;

    t1 = tbb::tick_count::now();
    std::vector<int> seq_a = shell_sort_sequential(a);
    t2 = tbb::tick_count::now();
    std::cout << "Sequential Time: " << (t2 - t1).seconds() << std::endl;

    t1 = tbb::tick_count::now();
    std::vector<int> parallel_a = shell_sort_parallel(a);
    t2 = tbb::tick_count::now();
    std::cout << "Parallel Time: " << (t2 - t1).seconds() << std::endl;

    ASSERT_EQ(seq_a, parallel_a);
}


TEST(Shell_sort_tbb, Test_big_size_int) {
    const int size_vector = 10000;
    std::vector<int> a = getRandomVectorInt(size_vector);
    tbb::tick_count t1, t2;

    t1 = tbb::tick_count::now();
    std::vector<int> seq_a = shell_sort_sequential(a);
    t2 = tbb::tick_count::now();
    std::cout << "Sequential Time: " << (t2 - t1).seconds() << std::endl;

    t1 = tbb::tick_count::now();
    std::vector<int> parallel_a = shell_sort_parallel(a);
    t2 = tbb::tick_count::now();
    std::cout << "Parallel Time: " << (t2 - t1).seconds() << std::endl;

    ASSERT_EQ(seq_a, parallel_a);
}

TEST(Shell_sort_tbb, Test_big_size_double) {
    const int size_vector = 10000;
    std::vector<double> a = getRandomVectorDouble(size_vector);
    tbb::tick_count t1, t2;

    t1 = tbb::tick_count::now();
    std::vector<double> seq_a = shell_sort_sequential(a);
    t2 = tbb::tick_count::now();
    std::cout << "Sequential Time: " << (t2 - t1).seconds() << std::endl;

    t1 = tbb::tick_count::now();
    std::vector<double> parallel_a = shell_sort_parallel(a);
    t2 = tbb::tick_count::now();
    std::cout << "Parallel Time: " << (t2 - t1).seconds() << std::endl;

    ASSERT_EQ(seq_a, parallel_a);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
