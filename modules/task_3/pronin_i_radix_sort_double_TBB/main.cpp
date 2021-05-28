// Copyright 2021 Pronin Igor
#include <gtest/gtest.h>
#include <tbb/tick_count.h>
#include <time.h>
#include <algorithm>
#include <vector>
#include "./pronin_i_radix_sort_double_TBB.h"

TEST(Radix_sort_double, Positive_numbers1000_) {
    int size = 10000000;
    int threads = 4;
    int left = 0;
    int right = 100;
    double* vec = new double[size];
    double* vec1 = new double[size];
    double* vec2 = new double[size];
    double* vec3 = new double[size];
    double* vec4 = new double[size];
    double* vec5 = new double[size];
    RandVec(vec, left, right, size);
    for (int i = 0; i < size; i++) {
        vec2[i] = vec[i];
        vec4[i] = vec[i];
    }
    // SortAll(vec4, vec5, size);
    tbb::tick_count t0 = tbb::tick_count::now();
    SeqSort(vec, vec1, size, threads);
    tbb::tick_count t1 = tbb::tick_count::now();
    std::cout << "SeqTime:" << (t1 - t0).seconds() << std::endl;
    tbb::tick_count t00 = tbb::tick_count::now();
    ParSort(vec2, vec3, size, threads);
    tbb::tick_count t11 = tbb::tick_count::now();
    std::cout << "ParTime:" << (t11 - t00).seconds() << std::endl;
    std::vector<double> v;
    std::vector<double> v2;
    for (int i = 0; i < size; i++) {
        v.push_back(vec1[i]);
        v2.push_back(vec3[i]);
    }
    ASSERT_EQ(v, v2);
    delete[] vec;
    delete[] vec1;
    delete[] vec2;
    delete[] vec3;
    delete[] vec4;
    delete[] vec5;
}
TEST(Radix_sort_double, Negative_numbers) {
    int size = 20000;
    int threads = 3;
    int left = -100;
    int right = -1;
    double* vec = new double[size];
    double* vec1 = new double[size];
    double* vec2 = new double[size];
    double* vec3 = new double[size];
    RandVec(vec, left, right, size);
    for (int i = 0; i < size; i++) {
        vec2[i] = vec[i];
    }
    SortAll(vec, vec1, size);
    ParSort(vec2, vec3, size, threads);
    std::vector<double> v;
    std::vector<double> v2;
    for (int i = 0; i < size; i++) {
        v.push_back(vec[i]);
        v2.push_back(vec3[i]);
    }
    ASSERT_EQ(v, v2);
    delete[] vec;
    delete[] vec1;
    delete[] vec2;
    delete[] vec3;
}
TEST(Radix_sort_double, All_numbers) {
    int size = 20000;
    int threads = 3;
    int left = -100;
    int right = 100;
    double* vec = new double[size];
    double* vec1 = new double[size];
    double* vec2 = new double[size];
    double* vec3 = new double[size];
    RandVec(vec, left, right, size);
    for (int i = 0; i < size; i++) {
        vec2[i] = vec[i];
    }
    SortAll(vec, vec1, size);
    ParSort(vec2, vec3, size, threads);
    std::vector<double> v;
    std::vector<double> v2;
    for (int i = 0; i < size; i++) {
        v.push_back(vec[i]);
        v2.push_back(vec3[i]);
    }
    ASSERT_EQ(v, v2);
    delete[] vec;
    delete[] vec1;
    delete[] vec2;
    delete[] vec3;
}
TEST(Radix_sort_double, 100numbers) {
    int size = 20000;
    int threads = 3;
    int left = -100;
    int right = 100;
    double* vec = new double[size];
    double* vec1 = new double[size];
    double* vec2 = new double[size];
    double* vec3 = new double[size];
    RandVec(vec, left, right, size);
    for (int i = 0; i < size; i++) {
        vec2[i] = vec[i];
    }
    SortAll(vec, vec1, size);
    ParSort(vec2, vec3, size, threads);
    std::vector<double> v;
    std::vector<double> v2;
    for (int i = 0; i < size; i++) {
        v.push_back(vec[i]);
        v2.push_back(vec3[i]);
    }
    ASSERT_EQ(v, v2);
    delete[] vec;
    delete[] vec1;
    delete[] vec2;
    delete[] vec3;
}
TEST(Radix_sort_double, 1000numbers) {
    int size = 20000;
    int threads = 3;
    int left = -100;
    int right = 100;
    double* vec = new double[size];
    double* vec1 = new double[size];
    double* vec2 = new double[size];
    double* vec3 = new double[size];
    RandVec(vec, left, right, size);
    for (int i = 0; i < size; i++) {
        vec2[i] = vec[i];
    }
    SortAll(vec, vec1, size);
    ParSort(vec2, vec3, size, threads);
    std::vector<double> v;
    std::vector<double> v2;
    for (int i = 0; i < size; i++) {
        v.push_back(vec[i]);
        v2.push_back(vec3[i]);
    }
    ASSERT_EQ(v, v2);
    delete[] vec;
    delete[] vec1;
    delete[] vec2;
    delete[] vec3;
}
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
