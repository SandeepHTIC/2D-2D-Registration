
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

std::pair<cv::Point2f, std::vector<cv::Point2f>> Center_detection(
    const std::vector<cv::Point2f> &centroid,
    int r,
    int c);
