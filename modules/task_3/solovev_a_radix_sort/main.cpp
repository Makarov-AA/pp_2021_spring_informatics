// Copyright 2021 Solovev Aleksandr
#include <gtest/gtest.h>
#include <tbb/tick_count.h>

#include "../../../modules/task_3/solovev_a_radix_sort/radix_sort.h"


TEST(Parallel, Test_Very_Micro_Length) {
    const int length = 10;
    int statusseq = 0;
    int statuspar = 0;
    std::vector<int> buffer1(length);
    statusseq = generateRandomArray(&buffer1, -100, 100);
    std::vector<int> buffer2 = buffer1;
    statusseq = RadixSort(&buffer1);
    statusseq = SortingCheck(&buffer1);
    statuspar = ParallelSortingTBB(&buffer2);
    statuspar = SortingCheck(&buffer2);

    ASSERT_EQ(statusseq, statuspar);
}


TEST(Parallel, Test_Very_Small_Length) {
    const int length = 100;
    int statusseq = 0;
    int statuspar = 0;
    std::vector<int> buffer1(length);
    statusseq = generateRandomArray(&buffer1, -1000, 1000);
    std::vector<int> buffer2 = buffer1;
    statusseq = RadixSort(&buffer1);
    statusseq = SortingCheck(&buffer1);
    statuspar = ParallelSortingTBB(&buffer2);
    statuspar = SortingCheck(&buffer2);

    ASSERT_EQ(statusseq, statuspar);
}

TEST(Parallel, Test_Small_length) {
    const int length = 500;
    int statusseq = 0;
    int statuspar = 0;
    std::vector<int> buffer1(length);
    statusseq = generateRandomArray(&buffer1, -10000, 10000);

    std::vector<int> buffer2 = buffer1;
    statusseq = RadixSort(&buffer1);
    statusseq = SortingCheck(&buffer1);
    statuspar = ParallelSortingTBB(&buffer2);
    statuspar = SortingCheck(&buffer2);

    ASSERT_EQ(statusseq, statuspar);
}

TEST(Parallel, Test_Medium_Length) {
    const int length = 1000;
    int statusseq = 0;
    int statuspar = 0;
    std::vector<int> buffer1(length);
    statusseq = generateRandomArray(&buffer1,  -1000, 1000);
    std::vector<int> buffer2 = buffer1;
    statusseq = RadixSort(&buffer1);
    statusseq = SortingCheck(&buffer1);

    statuspar = ParallelSortingTBB(&buffer2);
    statuspar = SortingCheck(&buffer2);


    ASSERT_EQ(statusseq, statuspar);
}

TEST(Senquential, Test_Large_Length) {
    const int length = 1500;
    int statusseq = 0;
    int statuspar = 0;
    std::vector<int> buffer1(length);
    statusseq = generateRandomArray(&buffer1,  -1000, 1000);
    std::vector<int> buffer2 = buffer1;

    statusseq = RadixSort(&buffer1);
    statusseq = SortingCheck(&buffer1);

    statuspar = ParallelSortingTBB(&buffer2);
    statuspar = SortingCheck(&buffer2);


    ASSERT_EQ(statusseq, statuspar);
}

TEST(Senquential, Test_Very_Large_Length) {
    const int length = 2000;
    int statusseq = 0;
    int statuspar = 0;
    std::vector<int> buffer1(length);
    statusseq = generateRandomArray(&buffer1,  -5000, 5000);
    std::vector<int> buffer2 = buffer1;
    statusseq = RadixSort(&buffer1);
    statusseq = SortingCheck(&buffer1);

    statuspar = ParallelSortingTBB(&buffer2);
    statuspar = SortingCheck(&buffer2);


    ASSERT_EQ(statusseq, statuspar);
}

// TEST(Seqvssspar, Test_Only_Positive) {
    // int n = 3000000;
    // double start = 0, finish = 0, time_seq = 0, time_par = 0;
    // std::vector<double> vec = createRandomVector(n);
    // std::vector<double> vec_copy = vec;
    // tbb::tick_count t1_seq = tbb::tick_count::now();
    // qSortSeq(&vec, 0, n - 1);
    // tbb::tick_count t2_seq = tbb::tick_count::now();
    // time_seq = (t2_seq - t1_seq).seconds();
    // std::cout << "Time sequential: " << time_seq << std::endl;
    // tbb::tick_count t1_par = tbb::tick_count::now();
    // qSortTbb(&vec_copy);
    // tbb::tick_count t2_par = tbb::tick_count::now();
    // time_par = (t2_par - t1_par).seconds();
    // std::cout << "Time parallel: " << time_par << std::endl;
    // std::cout << "Acceleration " << time_seq / time_par << std::endl;
//     const int length = 15000000;
//     int statusseq = 0;
//     int statuspar = 0;
//     std::vector<int> buffer1(length);
//     statusseq = generateRandomArray(&buffer1, -10000, 10000);
//     std::vector<int> buffer2 = buffer1;
//     tbb::tick_count t1_seq = tbb::tick_count::now();
//     statusseq = RadixSort(&buffer1);
//     statusseq = SortingCheck(&buffer1);
//     tbb::tick_count t2_seq = tbb::tick_count::now();
//     double time_seq = (t2_seq - t1_seq).seconds();
//     std::cout << "Time sequential: " << time_seq << std::endl;
//     tbb::tick_count t1_par = tbb::tick_count::now();
//     statuspar = ParallelSortingTBB(&buffer2);
//     tbb::tick_count t2_par = tbb::tick_count::now();
//     double time_par = (t2_par - t1_par).seconds();
//     std::cout << "Time parallel: " << time_par << std::endl;
//     std::cout << "Acceleration " << time_seq / time_par << std::endl;
//     ASSERT_EQ(buffer2, buffer1);
// }

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
