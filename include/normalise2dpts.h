#ifndef NORMALISE2DPTS_H
#define NORMALISE2DPTS_H

#include <Eigen/Dense>
#include <vector>
#include <limits>
#include <cmath>
#include <stdexcept>

// Normalise 2D homogeneous points to zero-mean and sqrt(2) average distance
// Input: 3xN matrix of homogeneous points
// Output: pair {T*pts, T}
inline std::pair<Eigen::MatrixXd, Eigen::Matrix3d> normalise2dpts(const Eigen::MatrixXd& pts) {
    if (pts.rows() != 3) {
        throw std::invalid_argument("pts must be 3xN");
    }
    Eigen::MatrixXd newpts = pts;
    std::vector<int> finiteind;
    finiteind.reserve(pts.cols());
    for (int i = 0; i < pts.cols(); ++i) {
        if (std::abs(pts(2, i)) > std::numeric_limits<double>::epsilon()) {
            finiteind.push_back(i);
        }
    }
    for (int idx : finiteind) {
        newpts(0, idx) /= newpts(2, idx);
        newpts(1, idx) /= newpts(2, idx);
        newpts(2, idx) = 1.0;
    }
    Eigen::Vector2d c = Eigen::Vector2d::Zero();
    for (int idx : finiteind) {
        c(0) += newpts(0, idx);
        c(1) += newpts(1, idx);
    }
    c /= finiteind.size();
    for (int idx : finiteind) {
        newpts(0, idx) -= c(0);
        newpts(1, idx) -= c(1);
    }
    double meandist = 0.0;
    for (int idx : finiteind) {
        meandist += std::sqrt(newpts(0, idx)*newpts(0, idx) + newpts(1, idx)*newpts(1, idx));
    }
    meandist /= finiteind.size();
    double scale = std::sqrt(2.0) / meandist;
    Eigen::Matrix3d T;
    T << scale, 0, -scale*c(0),
         0, scale, -scale*c(1),
         0, 0, 1;
    return {T * pts, T};
}

#endif // NORMALISE2DPTS_H