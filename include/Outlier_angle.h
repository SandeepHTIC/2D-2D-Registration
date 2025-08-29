#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

// Returns filtered ring points (T_R_B)
std::vector<cv::Point2f> Outlier_angle(
    const std::vector<cv::Point2f>& FRB,
    const cv::Point2f& Centre
);