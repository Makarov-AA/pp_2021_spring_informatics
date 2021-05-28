// Copyright 2021 Bulychev Vladislav
#include "../../../modules/task_4/bulychev_v_int_rec/int_rec_std.h"
#include "../../../3rdparty/unapproved/unapproved.h"

double Calculation_Seq(std::vector<double> a, std::vector<double> b,
    int n, double(*f)(std::vector<double>)) {
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
        result += f(p);
    }

    int t4 = size_a;
    double t5 = 1;
    for (int i = 0; i < t4; i++) {
        t5 = t5 * h[i];
    }

    result = result * t5;

    return result;
}

double Calculation_Std(std::vector<double> a, std::vector<double> b,
    int n, double(*f)(std::vector<double>)) {
    int t_num = std::thread::hardware_concurrency();
    std::vector<std::thread> t_v;
    int size_a = a.size();
    std::vector<double> h;
    double result = 0.0;
    std::vector <double> p(size_a);
    int num = pow(n, size_a);

    int d = num / t_num;
    int r = num % t_num;
    std::vector<int> e(t_num);
    int s = 0;

    for (int j = 0; j < t_num; ++j) {
        e[j] = d;
        if (r != 0) {
            r--;
            e[j]++;
        }
    }

    for (int i = 0; i < size_a; i++) {
        double t1 = b[i] - a[i];
        double t2 = t1 / n;
        h.push_back(t2);
    }

    auto calculation = [&](int s, int e) {
        double l_result = 0;
        for (int i = s; i < e; ++i) {
            for (int j = 0; j < size_a; j++) {
                p[j] = (i % n) * h[j] + a[j] + h[j] * 0.5;
            }
            l_result += f(p);
        }
        result += l_result;
    };

    for (int j = 0; j < t_num; ++j) {
        int tmp = e[j];
        int end1 = s + tmp;
        t_v.emplace_back(std::thread(calculation, s, end1));
        s += tmp;
        t_v[j].join();
    }

    int t4 = size_a;
    double t5 = 1;
    for (int i = 0; i < t4; i++) {
        t5 = t5 * h[i];
    }

    result = result * t5;

    return result;
}
