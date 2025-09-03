#ifndef DECOMPOSE_CAMERA_H
#define DECOMPOSE_CAMERA_H

#include <Eigen/Dense>
#include <tuple>
#include "rq3.h"

// Decompose 3x4 camera matrix into K, R, camera center, principal point, principal vector
inline std::tuple<Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector2d, Eigen::Vector3d>
decompose_camera(const Eigen::Matrix<double, 3, 4> &P)
{
    Eigen::Vector3d p1 = P.col(0);
    Eigen::Vector3d p2 = P.col(1);
    Eigen::Vector3d p3 = P.col(2);
    Eigen::Vector3d p4 = P.col(3);
    Eigen::Matrix3d M;
    M.col(0) = p1;
    M.col(1) = p2;
    M.col(2) = p3;
    Eigen::Vector3d m3 = M.row(2).transpose();
    Eigen::Matrix3d Cmat;
    Cmat.col(0) = p2;
    Cmat.col(1) = p3;
    Cmat.col(2) = p4;
    double X = Cmat.determinant();
    Cmat.col(0) = p1;
    Cmat.col(1) = p3;
    Cmat.col(2) = p4;
    double Y = -Cmat.determinant();
    Cmat.col(0) = p1;
    Cmat.col(1) = p2;
    Cmat.col(2) = p4;
    double Z = Cmat.determinant();
    Cmat.col(0) = p1;
    Cmat.col(1) = p2;
    Cmat.col(2) = p3;
    double T = -Cmat.determinant();
    Eigen::Vector4d Pc_h(X, Y, Z, T);
    Pc_h /= Pc_h(3);
    Eigen::Vector3d Pc = Pc_h.head<3>();
    Eigen::Vector3d pp_homogeneous = M * m3;
    pp_homogeneous /= pp_homogeneous(2);
    Eigen::Vector2d pp = pp_homogeneous.head<2>();
    Eigen::Vector3d pv = M.determinant() * m3;
    pv.normalize();
    auto [K, Rc_w] = rq3(M);
    return {K, Rc_w, Pc, pp, pv};
}

#endif // DECOMPOSE_CAMERA_H