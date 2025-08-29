#ifndef REGISTRATION_CHECK_H
#define REGISTRATION_CHECK_H

#include <Eigen/Dense>
#include <string>

double registration_check(
    const Eigen::Matrix<double, 3, 4>& P,
    const Eigen::MatrixXd& W_dist_pts,
    const std::string& position,
    double r,
    const std::string& path_d,
    int rewarp);

#endif // REGISTRATION_CHECK_H