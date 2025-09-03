#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

// Structure to hold ICP angle results
struct ICPAngleResult
{
    std::vector<cv::Point2f> fRPts;
    std::vector<cv::Point2f> fRPtsRem;
    std::vector<float> icpFidFin;
};

ICPAngleResult icp_angle(
    const std::vector<cv::Point2f> &first_ring_balls,
    const cv::Point2f &Centre,
    const std::vector<std::vector<float>> &icpFid,
    std::vector<std::vector<float>> &add_CMM_Pts);
