#ifndef PLATE12WITHICP_H
#define PLATE12WITHICP_H

#include <opencv2/opencv.hpp>
#include <vector>

struct PlateFiducials {
    std::vector<cv::Point2f> plate1Final;
    std::vector<cv::Point2f> plate2Final;
    std::vector<cv::Point2f> icpPlatFid;
};

PlateFiducials plate12withICP(const cv::Mat& blob_image,
                              const cv::Point2f& Centre,
                              const std::vector<std::vector<float>>& icpFid);

#endif // PLATE12WITHICP_H
