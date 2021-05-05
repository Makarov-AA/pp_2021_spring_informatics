// Copyright 2021 Panova Olga
#include <gtest/gtest.h>
#include <vector>
#include "./shell_b_sort.h"
TEST(Odd_Even_Shell, Division_Is_Correct) {
    std::vector<int> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<std::vector<int>> div_a = Division(a, 4);
    std::vector<std::vector<int>> b = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };
    ASSERT_EQ(div_a, b);
}
TEST(Odd_Even_Shell, Seq_And_Par_Sorts_Correct_And_Equal) {
    std::vector<int> a = { 10, 9, 18, 7, 6, -5, 1, 22 };
    std::vector<int> res_a = SequentialOddEvenShellSort(a, 4);
    std::vector<int> res_c = ParallelOddEvenShellSort(a, 4);
    std::vector<int> b = { -5, 1, 6, 7, 9, 10, 18, 22 };
    ASSERT_EQ(res_a, b);
    ASSERT_EQ(res_c, b);
}
TEST(Odd_Even_Shell, Vec_12345_Sort) {
    std::vector<int> a = CreateVector(12345);
    std::vector<int> b = ParallelOddEvenShellSort(a, 4);
    std::sort(a.begin(), a.end());
    ASSERT_EQ(a, b);
}
TEST(Odd_Even_Shell, Vec_100000_Par_Sort) {
    std::vector<int> a = CreateVector(100000);
    std::vector<int> b = ParallelOddEvenShellSort(a, 4);
    std::sort(a.begin(), a.end());
    ASSERT_EQ(a, b);
}
TEST(Odd_Even_Shell, Vec_100000_Seq_Sort) {
    std::vector<int> a = CreateVector(100000);
    std::vector<int> b = SequentialOddEvenShellSort(a, 4);
    std::sort(a.begin(), a.end());
    ASSERT_EQ(a, b);
}
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
