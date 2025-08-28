#ifndef PLATE12ICP_H
#define PLATE12ICP_H

#include <opencv2/core.hpp>
#include <vector>

// Struct to hold a labeled 2D point
// Now using float for x and y to match cv::Point2f
struct LabeledPoint {
    float x, y;
    int label;
};

// Struct to hold final fiducials from ICP clustering
// Using LabeledPoint so we can store both coordinates and labels
struct PlateFiducials {
    std::vector<LabeledPoint> final_plate1;
    std::vector<LabeledPoint> final_plate2;
    std::vector<LabeledPoint> icpPlatFid; // optional, can be filled if needed
};

// Function declaration for ICP-based plate fiducial extraction
PlateFiducials plate12icp(const cv::Mat& blob_image,
                          const cv::Point2f& Centre,
                          const std::vector<cv::Point2f>& icpFid);

#endif // PLATE12ICP_H