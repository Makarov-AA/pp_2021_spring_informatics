// Copyright 2021 Makarov Alexander
#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "./image_convex_hull.h"

TEST(Components, Test_Snow) {
    const std::vector<int> image = {
        0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
        1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1,
        1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1,
        1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1,
        1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1,
        1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1,
        1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1,
        1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1,
        0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
    };
    int h = 13, w = 13;
	
	/*cv::Mat binImg(h, w, CV_8U);
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			binImg.at<int>(i, j) = (image[i * w + j] == 0) ? 0 : 255;
		}
	}
	cv::imshow("Original image", binImg);
	int k = cv::waitKey(0);*/
	
    std::vector<int> result = mark_components(image, w, h);
    for (int i = 0; i < h; i++){
        for (int j = 0; j < w; j++)
            std::cout << result[i * w + j] << " ";
        std::cout << std::endl;
    }

    std::vector<std::list <std::pair<int, int> > > convex_hulls = get_convex_hulls(result, w, h);
    for (int i = 0; i < convex_hulls.size(); i++) {
        std::cout << i + 2 << ": ";
        for (auto point : convex_hulls[i]) {
            std::cout << "(" << point.first << ";" << point.second << ") ";
        }
        std::cout << std::endl;
    }
    ASSERT_NO_THROW(result = mark_components(image, w, h));
}

TEST(Components, Test_Fat_Snow) {
    const std::vector<int> image = {
        0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0,
        1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1,
        1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1,
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
        1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1,
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
        1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1,
        1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1,
        0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0,
    };
    int h = 13, w = 13;
    std::vector<int> result = mark_components(image, w, h);
    for (int i = 0; i < h; i++){
        for (int j = 0; j < w; j++)
            std::cout << result[i * w + j] << " ";
        std::cout << std::endl;
    }   
    
    std::vector<std::list <std::pair<int, int> > > convex_hulls = get_convex_hulls(result, w, h);
    for (int i = 0; i < convex_hulls.size(); i++) {
        std::cout << i + 2 << ": ";
        for (auto point : convex_hulls[i]) {
            std::cout << "(" << point.first << ";" << point.second << ") ";
        }
        std::cout << std::endl;
    }
    ASSERT_NO_THROW(result = mark_components(image, w, h));
}

TEST(Components, Test_4_Plus) {
    const std::vector<int> image = {
        1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
        1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
        0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
        1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
        1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
        1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
        1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
        1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
        0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
        1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
        1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
    };
    int h = 13, w = 13;
    std::vector<int> result = mark_components(image, w, h);
    for (int i = 0; i < h; i++){
        for (int j = 0; j < w; j++)
            std::cout << result[i * w + j] << " ";
        std::cout << std::endl;
    }
    
    std::vector<std::list <std::pair<int, int> > > convex_hulls = get_convex_hulls(result, w, h);
    for (int i = 0; i < convex_hulls.size(); i++) {
        std::cout << i + 2 << ": ";
        for (auto point : convex_hulls[i]) {
            std::cout << "(" << point.first << ";" << point.second << ") ";
        }
        std::cout << std::endl;
    }
    ASSERT_NO_THROW(result = mark_components(image, w, h));
}

TEST(Components, Test_4_Squares) {
    const std::vector<int> image = {
        0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0,
        0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0,
        0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
        0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1,
        1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1,
        1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0,
        1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1,
        0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
        1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0,
        0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0,
    };
    int h = 13, w = 13;
    std::vector<int> result = mark_components(image, w, h);
    for (int i = 0; i < h; i++){
        for (int j = 0; j < w; j++)
            std::cout << result[i * w + j] << " ";
        std::cout << std::endl;
    }

    std::vector<std::list <std::pair<int, int> > > convex_hulls = get_convex_hulls(result, w, h);
    for (int i = 0; i < convex_hulls.size(); i++) {
        std::cout << i + 2 << ": ";
        for (auto point : convex_hulls[i]) {
            std::cout << "(" << point.first << ";" << point.second << ") ";
        }
        std::cout << std::endl;
    }
    ASSERT_NO_THROW(result = mark_components(image, w, h));
}

TEST(Components, Test_Perim_sqres) {
    const std::vector<int> image = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0,
        0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0,
        0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0,
        0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0,
        0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    int h = 13, w = 13;
    std::vector<int> result = mark_components(image, w, h);
    for (int i = 0; i < h; i++){
        for (int j = 0; j < w; j++)
            std::cout << result[i * w + j] << " ";
        std::cout << std::endl;
    }
    
    std::vector<std::list <std::pair<int, int> > > convex_hulls = get_convex_hulls(result, w, h);
    for (int i = 0; i < convex_hulls.size(); i++) {
        std::cout << i + 2 << ": ";
        for (auto point : convex_hulls[i]) {
            std::cout << "(" << point.first << ";" << point.second << ") ";
        }
        std::cout << std::endl;
    }
    ASSERT_NO_THROW(result = mark_components(image, w, h));
}

TEST(Components, Test_Perim_no_angles) {
    const std::vector<int> image = {
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0,
        0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0,
        0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0,
        0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0,
        0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    };
    int h = 13, w = 13;
    std::vector<int> result = mark_components(image, w, h);
    for (int i = 0; i < h; i++){
        for (int j = 0; j < w; j++)
            std::cout << result[i * w + j] << " ";
        std::cout << std::endl;
    }
    
    std::vector<std::list <std::pair<int, int> > > convex_hulls = get_convex_hulls(result, w, h);
    for (int i = 0; i < convex_hulls.size(); i++) {
        std::cout << i + 2 << ": ";
        for (auto point : convex_hulls[i]) {
            std::cout << "(" << point.first << ";" << point.second << ") ";
        }
        std::cout << std::endl;
    }
    ASSERT_NO_THROW(result = mark_components(image, w, h));
}

TEST(Components, Test_My_Image) {
    cv::Mat read_image = cv::imread("D:/Pictures/GrayWorld.jpg", cv::IMREAD_GRAYSCALE );
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
    std::vector<std::list <std::pair<int, int> > > convex_hulls = get_convex_hulls(result, w, h);
	cv::Mat result_image(binary_image.size(), CV_8UC3);
	cv::cvtColor(binary_image, result_image, cv::COLOR_GRAY2BGR);
	for (auto convex_hull : convex_hulls) {
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
    ASSERT_NO_THROW(result = mark_components(image, w, h));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
