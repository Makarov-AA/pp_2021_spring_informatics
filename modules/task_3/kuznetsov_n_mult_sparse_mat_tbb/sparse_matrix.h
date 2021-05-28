// Copyright 2021 Kuznetsov Nikita
#ifndef MODULES_TASK_3_KUZNETSOV_N_MULT_SPARSE_MAT_TBB_SPARSE_MATRIX_H_
#define MODULES_TASK_3_KUZNETSOV_N_MULT_SPARSE_MAT_TBB_SPARSE_MATRIX_H_
#include <vector>

struct CcsMatrix {
    int M, N;
    int not_zero_number;

    std::vector<double> value;
    std::vector<int> row;
    std::vector<int> colIndex;

    CcsMatrix(int _M, int _N, int nz);
    CcsMatrix() = default;
};

double measurementOfTime(int M1, int N1, int N2, int num_threads);
bool operator==(const CcsMatrix& m1, const CcsMatrix& m2);
CcsMatrix generateMatrix(int M, int N);
CcsMatrix transposeMatrix(const CcsMatrix* m);
double scalarMultiplication(const CcsMatrix* transposed_m, const CcsMatrix* m,
    int i, int j);
CcsMatrix matrixMultiplicate(const CcsMatrix* m1, const CcsMatrix* m2);

#endif  // MODULES_TASK_3_KUZNETSOV_N_MULT_SPARSE_MAT_TBB_SPARSE_MATRIX_H_
