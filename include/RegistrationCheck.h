
#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <string>

// Computes registration error between projected 3D points and detected blob centroids.
// Returns the chosen error metric in pixels (lower is better).
// - Uses mutual nearest-neighbor threshold sweep and Hungarian assignment.
// - Logs concise error summary to std::cout.
double registrationCheck(
    const Eigen::Matrix<double, 3, 4>& P,
    const Eigen::MatrixXd& W_dist_pts,
    const std::string& position,
    double r,
    const std::string& path_d,
    int rewarp);
