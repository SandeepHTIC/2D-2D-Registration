#ifndef ESTIMATE_CAMERA_MATRIX_H
#define ESTIMATE_CAMERA_MATRIX_H

#include <Eigen/Dense>
#include <vector>
#include <utility>

std::pair<Eigen::Matrix<double, 3, 4>, std::vector<double>> estimate_camera_matrix(
    const Eigen::MatrixXd& imagePoints,
    const Eigen::MatrixXd& worldPoints);

#endif // ESTIMATE_CAMERA_MATRIX_H