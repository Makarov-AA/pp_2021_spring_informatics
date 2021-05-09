// Copyright 2021 Makarov Alexander

#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include <random>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "tbb/tick_count.h"
#include "./image_convex_hull.h"

const int prm_size = 15;
std::vector<std::vector<int> > primitives = {
    {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    },
    {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    },
    {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
        1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
        1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
        1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
        1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
        1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
        1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
        1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
        1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
        1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
        1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    },
    {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
        1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    },
    {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1,
        1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1,
        1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1,
        1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
        1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1,
        1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
        1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1,
        1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1,
        1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    },
};

std::vector<std::list <std::pair<int, int> > > primitives_convex_hulls = {
    {
        {1, 6}, {1, 7}, {1, 8}, {6, 13}, {7, 13}, {8, 13}, {13, 8}, {13, 6},
        {8, 1}, {6, 1}
    },
    {
        {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {1, 7}, {1, 8}, {1, 9},
        {1, 10}, {1, 11}, {1, 12}, {1, 13}, {2, 13}, {3, 13}, {4, 13}, {5, 13},
        {6, 13}, {7, 13}, {8, 13}, {9, 13}, {10, 13}, {11, 13}, {12, 13},
        {13, 13}
    },
    {
        {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {1, 7}, {1, 8}, {1, 9},
        {1, 10}, {1, 11}, {1, 12}, {1, 13}, {2, 13}, {3, 13}, {4, 13}, {5, 13},
        {6, 13}, {7, 13}, {8, 13}, {9, 13}, {10, 13}, {11, 13}, {12, 13},
        {13, 13}, {13, 1}
    },
    {
        {1, 4}, {1, 5}, {1, 6}, {1, 7}, {1, 8}, {1, 9}, {1, 10}, {2, 11},
        {3, 12}, {4, 13}, {5, 13}, {6, 13}, {7, 13}, {8, 13}, {9, 13},
        {10, 13}, {13, 10}, {13, 4}, {10, 1}, {4, 1}, {3, 2}, {2, 3}
    },
    {
        {1, 1}, {1, 2}, {1, 12}, {1, 13}, {2, 13}, {12, 13}, {13, 13}, {13, 1}
    },
};

TEST(Components, Test_Vert_Cross) {
    const std::vector<int> image = primitives[0];
    int h = 15, w = 15;
    std::vector<int> marked_image = mark_components(image, w, h);
    /*for (int i = 0; i < h; i++){
        for (int j = 0; j < w; j++)
            std::cout << marked_image[i * w + j] << " ";
        std::cout << std::endl;
    }*/

    std::vector<std::list <std::pair<int, int> > > convex_hulls =
                             get_convex_hulls(marked_image, w, h);
    /*for (int i = 0; i < convex_hulls.size(); i++) {
        std::cout << i + 2 << ": ";
        for (auto point : convex_hulls[i]) {
            std::cout << "(" << point.first << ";" << point.second << ") ";
        }
        std::cout << std::endl;
    }*/
    std::vector<std::list <std::pair<int, int> > > expected_hulls;
    expected_hulls.push_back(primitives_convex_hulls[0]);
    ASSERT_EQ(convex_hulls, expected_hulls);
}

TEST(Components, Test_Triangle) {
    const std::vector<int> image = primitives[1];
    int h = 15, w = 15;
    std::vector<int> marked_image = mark_components(image, w, h);
    std::vector<std::list <std::pair<int, int> > > convex_hulls =
                             get_convex_hulls(marked_image, w, h);
    std::vector<std::list <std::pair<int, int> > > expected_hulls;
    expected_hulls.push_back(primitives_convex_hulls[1]);
    ASSERT_EQ(convex_hulls, expected_hulls);
}

TEST(Components, Test_Perimeter) {
    const std::vector<int> image = primitives[2];
    int h = 15, w = 15;
    std::vector<int> marked_image = mark_components(image, w, h);
    std::vector<std::list <std::pair<int, int> > > convex_hulls =
                             get_convex_hulls(marked_image, w, h);
    std::vector<std::list <std::pair<int, int> > > expected_hulls;
    expected_hulls.push_back(primitives_convex_hulls[2]);
    ASSERT_EQ(convex_hulls, expected_hulls);
}

TEST(Components, Test_Sqr_Without_Angles) {
    const std::vector<int> image = primitives[3];
    int h = 15, w = 15;
    std::vector<int> marked_image = mark_components(image, w, h);
    std::vector<std::list <std::pair<int, int> > > convex_hulls =
                             get_convex_hulls(marked_image, w, h);
    std::vector<std::list <std::pair<int, int> > > expected_hulls;
    expected_hulls.push_back(primitives_convex_hulls[3]);
    ASSERT_EQ(convex_hulls, expected_hulls);
}

TEST(Components, Test_Diag_Cross) {
    const std::vector<int> image = primitives[4];
    int h = 15, w = 15;
    std::vector<int> marked_image = mark_components(image, w, h);
    std::vector<std::list <std::pair<int, int> > > convex_hulls =
                            get_convex_hulls(marked_image, w, h);
    std::vector<std::list <std::pair<int, int> > > expected_hulls;
    expected_hulls.push_back(primitives_convex_hulls[4]);
    ASSERT_EQ(convex_hulls, expected_hulls);
}

TEST(Components, Test_100x100_prim_image) {
    int h = 100, w = 100;
    int size = w * h;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<int> image(size * prm_size * prm_size);
    std::vector<std::list <std::pair<int, int> > > expected_hulls(size);
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
            int prm_num = static_cast<int>(gen() % primitives.size());
            expected_hulls[i * w + j] = primitives_convex_hulls[prm_num];
            for (std::pair<int, int>& point : expected_hulls[i * w + j]) {
                point.first += j * prm_size;
                point.second += i * prm_size;
            }
            for (int k = 0; k < prm_size; k++)
                for (int q = 0; q < prm_size; q++) {
                    int idx = (i * w * prm_size + j) * prm_size +
                           k * w * prm_size + q;
                    image[idx] = primitives[prm_num][k * prm_size + q];
                }
        }
    std::vector<int> marked_image = mark_components(image, w * prm_size,
                                                           h * prm_size);
    std::vector<std::list <std::pair<int, int> > > convex_hulls_seq,
                                                   convex_hulls_par;

    tbb::tick_count start_time, end_time;
    tbb::tick_count::interval_t seq_time, par_time;

    start_time = tbb::tick_count::now();
    convex_hulls_seq = get_convex_hulls_seq(marked_image, w * prm_size,
                                                          h * prm_size);
    end_time = tbb::tick_count::now();
    seq_time = end_time - start_time;

    start_time = tbb::tick_count::now();
    convex_hulls_par = get_convex_hulls(marked_image, w * prm_size,
                                                      h * prm_size);
    end_time = tbb::tick_count::now();
    par_time = end_time - start_time;

    std::cout << "Seq time: " << seq_time.seconds() << " s" << std::endl;
    std::cout << "Par time: " << par_time.seconds() << " s" << std::endl;

    ASSERT_EQ(convex_hulls_par, convex_hulls_seq);
    ASSERT_EQ(convex_hulls_par, expected_hulls);
}


TEST(Components, Test_200x200_prim_image) {
    int h = 200, w = 200;
    int size = w * h;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<int> image(size * prm_size * prm_size);
    std::vector<std::list <std::pair<int, int> > > expected_hulls(size);
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
            int prm_num = static_cast<int>(gen() % primitives.size());
            expected_hulls[i * w + j] = primitives_convex_hulls[prm_num];
            for (std::pair<int, int>& point : expected_hulls[i * w + j]) {
                point.first += j * prm_size;
                point.second += i * prm_size;
            }
            for (int k = 0; k < prm_size; k++)
                for (int q = 0; q < prm_size; q++) {
                    int idx = (i * w * prm_size + j) * prm_size +
                           k * w * prm_size + q;
                    image[idx] = primitives[prm_num][k * prm_size + q];
                }
        }
    std::vector<int> marked_image = mark_components(image, w * prm_size,
                                                           h * prm_size);
    std::vector<std::list <std::pair<int, int> > > convex_hulls_seq,
                                                   convex_hulls_par;

    tbb::tick_count start_time, end_time;
    tbb::tick_count::interval_t seq_time, par_time;

    start_time = tbb::tick_count::now();
    convex_hulls_seq = get_convex_hulls_seq(marked_image, w * prm_size,
                                                          h * prm_size);
    end_time = tbb::tick_count::now();
    seq_time = end_time - start_time;

    start_time = tbb::tick_count::now();
    convex_hulls_par = get_convex_hulls(marked_image, w * prm_size,
                                                      h * prm_size);
    end_time = tbb::tick_count::now();
    par_time = end_time - start_time;

    std::cout << "Seq time: " << seq_time.seconds() << " s" << std::endl;
    std::cout << "Par time: " << par_time.seconds() << " s" << std::endl;

    ASSERT_EQ(convex_hulls_par, convex_hulls_seq);
    ASSERT_EQ(convex_hulls_par, expected_hulls);
}

TEST(Components, Test_700x700_worst) {
    int h = 700, w = 700;
    int size = w * h;
    std::vector<int> image(size, 0);
    std::vector<std::list <std::pair<int, int> > > expected_hulls(1);
    for (int i = 0; i < h; i++) {
        expected_hulls[0].push_back(std::pair<int, int>(0, i));
    }
    for (int i = 1; i < w; i++) {
        expected_hulls[0].push_back(std::pair<int, int>(i, h - 1));
    }
    expected_hulls[0].push_back(std::pair<int, int>(w - 1, 0));
    std::vector<int> marked_image = mark_components(image, w, h);
    std::vector<std::list <std::pair<int, int> > > convex_hulls_seq,
                                                   convex_hulls_par;

    tbb::tick_count start_time, end_time;
    tbb::tick_count::interval_t seq_time, par_time;

    start_time = tbb::tick_count::now();
    convex_hulls_seq = get_convex_hulls_seq(marked_image, w, h);
    end_time = tbb::tick_count::now();
    seq_time = end_time - start_time;

    start_time = tbb::tick_count::now();
    convex_hulls_par = get_convex_hulls(marked_image, w, h);
    end_time = tbb::tick_count::now();
    par_time = end_time - start_time;

    std::cout << "Seq time: " << seq_time.seconds() << " s" << std::endl;
    std::cout << "Par time: " << par_time.seconds() << " s" << std::endl;

    ASSERT_EQ(convex_hulls_par, convex_hulls_seq);
    ASSERT_EQ(convex_hulls_par, expected_hulls);
}

TEST(Components, Test_My_Image) {
    cv::Mat read_image = cv::imread("D:/Pictures/GrayWorld.jpg", cv::IMREAD_GRAYSCALE);
	cv::imshow("Original", read_image);
	int k = cv::waitKey(0);
	cv::Mat binary_image(read_image.size(), read_image.type());
	cv::threshold(read_image, binary_image, 100, 255, cv::THRESH_BINARY_INV);
	cv::imshow("Binary", binary_image);
	cv::waitKey(0);
	int w = binary_image.size().width;
	int h = binary_image.size().height;
	std::vector<int> image(w * h);
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			image[i * w + j] = (binary_image.at<uchar>(i, j) == 0) ? 0 : 1;
		}
	}
	cv::Mat marked_image(binary_image.size(), CV_8UC3);
    std::vector<int> result = mark_components(image, w, h);
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			int mark = result[i * w + j];
			cv::Vec3b color;
			if (mark == 1) {
				color[0] = 255;
				color[1] = 255;
				color[2] = 255;
			} else {
				color[0] = mark % 2 * 200;
				color[1] = mark % 3 * 100;
				color[2] = mark * 100 % 255;
			}
			marked_image.at<cv::Vec3b>(i, j) = color;
		}
	}
	cv::imshow("Marks", marked_image);
	cv::waitKey(0);
    
    std::vector<std::list <std::pair<int, int> > > convex_hulls_seq,
                                                   convex_hulls_par;
    tbb::tick_count start_time, end_time;
    tbb::tick_count::interval_t seq_time, par_time;
    
    start_time = tbb::tick_count::now();
    convex_hulls_seq = get_convex_hulls_seq(result, w, h);
    end_time = tbb::tick_count::now();
    seq_time = end_time - start_time;

    start_time = tbb::tick_count::now();
    convex_hulls_par = get_convex_hulls(result, w, h);
    end_time = tbb::tick_count::now();
    par_time = end_time - start_time;
    
    std::cout << "Seq time: " << seq_time.seconds() << " s" << std::endl;
    std::cout << "Par time: " << par_time.seconds() << " s" << std::endl;
    
	cv::Mat result_image(binary_image.size(), CV_8UC3);
	cv::cvtColor(binary_image, result_image, cv::COLOR_GRAY2BGR);
	for (auto convex_hull : convex_hulls_par) {
		std::pair<int, int> first_point = convex_hull.front();
		for (auto iter = ++convex_hull.begin(); iter != convex_hull.end(); iter++) {
			std::pair<int, int> second_point = *iter;
			cv::line(result_image, cv::Point(first_point.first, first_point.second),
			         cv::Point(second_point.first, second_point.second),
					 cv::Scalar(0, 255, 0));
			first_point = second_point;
		}
		cv::line(result_image, cv::Point(convex_hull.front().first, convex_hull.front().second),
			     cv::Point(convex_hull.back().first, convex_hull.back().second),
			     cv::Scalar(0, 255, 0));
	}
	cv::imshow("Result", result_image);
	cv::waitKey(0);
    ASSERT_EQ(convex_hulls_par, convex_hulls_seq);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
