#ifndef CALIBRATION_H
#define CALIBRATION_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>

struct CalibrationResult {
    cv::Mat resultMatrix;    // 4x4 homogeneous transformation matrix
    double RPE;              // Registration Projection Error
};

/**
 * @brief Main calibration function that estimates camera projection matrix
 * 
 * @param position      Camera position identifier (e.g., "AP", "LAT")
 * @param C2R          Camera-to-reference transformation (7-element vector: [tx,ty,tz,qx,qy,qz,qw])
 * @param C2D          Camera-to-detector transformation (7-element vector: [tx,ty,tz,qx,qy,qz,qw])
 * @param W            World coordinate points (Nx3 matrix)
 * @param js           Joint space parameters (currently unused)
 * @param Dpts         Detector points (Mx3 matrix)
 * @param r            Image height for coordinate system conversion
 * @param path_d       Base path for data files and outputs
 * @param rewarp       Flag for rewarping mode (0=normal, 1=rewarp)
 * @return CalibrationResult containing the 4x4 transformation matrix and RPE
 */
CalibrationResult Calibration(
    const std::string& position,
    const cv::Mat& C2R,
    const cv::Mat& C2D,
    const cv::Mat& W,
    const cv::Mat& js,
    const cv::Mat& Dpts,
    double r,
    const std::string& path_d,
    int rewarp
);

RefConversionResult Ref_conversion(
    const cv::Mat& C2R,
    const cv::Mat& C2D,
    const cv::Mat& W,
    const cv::Mat& Dpts
);

cv::Mat estimateCameraMatrix(
    const cv::Mat& imagePoints,
    const cv::Mat& worldPoints,
    cv::Mat& reprojectionErrors
);


#endif // CALIBRATION_H