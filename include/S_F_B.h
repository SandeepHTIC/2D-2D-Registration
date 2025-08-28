#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

// Generate symmetric ring points every 45 degrees around center
std::vector<cv::Point2f> S_F_B(
    const cv::Point2f& ball,
    const cv::Point2f& C
);
