#ifndef RQ3_H
#define RQ3_H

#include <Eigen/Dense>

// RQ decomposition for 3x3 matrix (similar to MATLAB rq3)
// Returns pair {R (upper triangular), Q (orthonormal)} such that A = R*Q
inline std::pair<Eigen::Matrix3d, Eigen::Matrix3d> rq3(const Eigen::Matrix3d &A_in)
{
    Eigen::Matrix3d A = A_in;
    double eps = 1e-10;
    // Step 1: Qx to zero out A(2,1)
    A(2, 2) += eps;
    double c = -A(2, 2) / std::sqrt(A(2, 2) * A(2, 2) + A(2, 1) * A(2, 1));
    double s = A(2, 1) / std::sqrt(A(2, 2) * A(2, 2) + A(2, 1) * A(2, 1));
    Eigen::Matrix3d Qx = Eigen::Matrix3d::Identity();
    Qx(1, 1) = c;
    Qx(1, 2) = -s;
    Qx(2, 1) = s;
    Qx(2, 2) = c;
    Eigen::Matrix3d R = A * Qx;
    // Step 2: Qy to zero out R(2,0)
    R(2, 2) += eps;
    c = R(2, 2) / std::sqrt(R(2, 2) * R(2, 2) + R(2, 0) * R(2, 0));
    s = R(2, 0) / std::sqrt(R(2, 2) * R(2, 2) + R(2, 0) * R(2, 0));
    Eigen::Matrix3d Qy = Eigen::Matrix3d::Identity();
    Qy(0, 0) = c;
    Qy(0, 2) = s;
    Qy(2, 0) = -s;
    Qy(2, 2) = c;
    R = R * Qy;
    // Step 3: Qz to zero out R(1,0)
    R(1, 1) += eps;
    c = -R(1, 1) / std::sqrt(R(1, 1) * R(1, 1) + R(1, 0) * R(1, 0));
    s = R(1, 0) / std::sqrt(R(1, 1) * R(1, 1) + R(1, 0) * R(1, 0));
    Eigen::Matrix3d Qz = Eigen::Matrix3d::Identity();
    Qz(0, 0) = c;
    Qz(0, 1) = -s;
    Qz(1, 0) = s;
    Qz(1, 1) = c;
    R = R * Qz;
    // Accumulate Q
    Eigen::Matrix3d Q = Qz.transpose() * Qy.transpose() * Qx.transpose();
    // Make diagonal of R positive
    for (int n = 0; n < 3; ++n)
    {
        if (R(n, n) < 0)
        {
            R.col(n) *= -1;
            Q.row(n) *= -1;
        }
    }
    return {R, Q};
}

#endif // RQ3_H