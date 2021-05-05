// Copyright 2021 Bulychev Vladislav
#include <omp.h>
#include <vector>
#include <numeric>
#include <utility>
#include <cmath>
#include <functional>
#include "../../../modules/task_2/bulychev_v_int_rec/int_rec_omp.h"

double Calculation_Seq(std::vector<double> a, std::vector<double> b,
    int n, double(*f)(double, double)) {
    int size_a = a.size();
    std::vector<double> h;
    double result = 0.0;
    std::vector <double> p(size_a);
    int num = pow(n, size_a);

    for (int i = 0; i < size_a; i++) {
        double t1 = b[i] - a[i];
        double t2 = t1 / n;
        h.push_back(t2);
    }

    for (int i = 0; i < num; i++) {
        for (int j = 0; j < size_a; j++) {
            double t3 = h[j] * 0.5;
            p[j] = (i % n) * h[j] + a[j] + t3;
        }
        result += f(p[0], p[1]);
    }

    int t4 = size_a;
    double t5 = 1;
    for (int i = 0; i < t4; i++) {
        t5 = t5 * h[i];
    }

    result = result * t5;

    return result;
}

double Calculation_Omp(std::vector<double> a, std::vector<double> b,
    int n, double(*f)(double, double)) {
    int const threads = 4;
    omp_set_num_threads(threads);

    int size_a = a.size();
    std::vector<double> h;
    int num = pow(n, size_a);

    for (int i = 0; i < size_a; i++) {
        double t1 = b[i] - a[i];
        double t2 = t1 / n;
        h.push_back(t2);
    }

    double result = 0.0;
    std::vector <double> p(size_a);

#pragma omp parallel for reduction(+ : result)
    for (int i = 0; i < num; i++) {
        for (int j = 0; j < size_a; j++) {
            p[j] = (i % n) * h[j] + a[j] + (h[j] * 0.5);
        }
        result += f(p[0], p[1]);
    }

    int t4 = size_a;
    double t5 = 1;
    for (int i = 0; i < t4; i++) {
        t5 = t5 * h[i];
    }

    result = result * t5;

    return result;
}
