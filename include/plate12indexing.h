#ifndef ICP_SECTION_CONVERT_H
#define ICP_SECTION_CONVERT_H

#include "plate12icp.h"        // brings in LabeledPoint & PlateFiducials
#include <vector>
#include <opencv2/core.hpp>    // for cv::Point2d

// Forward declaration only — no function body in the header
// This function will take an existing PlateFiducials, apply ICP post-processing,
// and return a new PlateFiducials with updated points.
PlateFiducials plate12withICP_post(
    const PlateFiducials& plateIn,
    const std::vector<std::vector<float>>& Z3,
    const cv::Point2d& Centre
);

#endif // ICP_SECTION_CONVERT_H