#ifndef DECOMPOSE_CAMERA_H
#define DECOMPOSE_CAMERA_H

#include <Eigen/Dense>
#include <tuple>

std::tuple<Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector2d, Eigen::Vector3d>
 decompose_camera(const Eigen::Matrix<double, 3, 4>& P);

// RQ decomposition helper used by decompose_camera
std::pair<Eigen::Matrix3d, Eigen::Matrix3d> rq3_decompose(const Eigen::Matrix3d& A);

#endif // DECOMPOSE_CAMERA_H