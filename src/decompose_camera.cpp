#include "decompose_camera.h"
#include <cmath>

static std::pair<Eigen::Matrix3d, Eigen::Matrix3d> rq3_impl(const Eigen::Matrix3d& A_in) {
    Eigen::Matrix3d A = A_in;
    double eps = 1e-10;
    // Step 1: Qx to zero out A(2,1)
    A(2,2) += eps;
    double c = -A(2,2) / std::sqrt(A(2,2)*A(2,2) + A(2,1)*A(2,1));
    double s =  A(2,1) / std::sqrt(A(2,2)*A(2,2) + A(2,1)*A(2,1));
    Eigen::Matrix3d Qx = Eigen::Matrix3d::Identity();
    Qx(1,1) = c; Qx(1,2) = -s;
    Qx(2,1) = s; Qx(2,2) =  c;
    Eigen::Matrix3d R = A * Qx;
    // Step 2: Qy to zero out R(2,0)
    R(2,2) += eps;
    c = R(2,2) / std::sqrt(R(2,2)*R(2,2) + R(2,0)*R(2,0));
    s = R(2,0) / std::sqrt(R(2,2)*R(2,2) + R(2,0)*R(2,0));
    Eigen::Matrix3d Qy = Eigen::Matrix3d::Identity();
    Qy(0,0) = c; Qy(0,2) = s;
    Qy(2,0) = -s; Qy(2,2) = c;
    R = R * Qy;
    // Step 3: Qz to zero out R(1,0)
    R(1,1) += eps;
    c = -R(1,1) / std::sqrt(R(1,1)*R(1,1) + R(1,0)*R(1,0));
    s =  R(1,0) / std::sqrt(R(1,1)*R(1,1) + R(1,0)*R(1,0));
    Eigen::Matrix3d Qz = Eigen::Matrix3d::Identity();
    Qz(0,0) = c; Qz(0,1) = -s;
    Qz(1,0) = s; Qz(1,1) =  c;
    R = R * Qz;
    // Accumulate Q
    Eigen::Matrix3d Q = Qz.transpose() * Qy.transpose() * Qx.transpose();
    // Make diagonal of R positive
    for (int n=0; n<3; ++n) {
        if (R(n,n) < 0) {
            R.col(n) *= -1;
            Q.row(n) *= -1;
        }
    }
    return {R, Q};
}

std::pair<Eigen::Matrix3d, Eigen::Matrix3d> rq3_decompose(const Eigen::Matrix3d& A) {
    return rq3_impl(A);
}

std::tuple<Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector2d, Eigen::Vector3d>
 decompose_camera(const Eigen::Matrix<double, 3, 4>& P)
{
    // Extract columns
    Eigen::Vector3d p1 = P.col(0);
    Eigen::Vector3d p2 = P.col(1);
    Eigen::Vector3d p3 = P.col(2);
    Eigen::Vector3d p4 = P.col(3);

    Eigen::Matrix3d M;
    M.col(0) = p1;
    M.col(1) = p2;
    M.col(2) = p3;
    Eigen::Vector3d m3 = M.row(2).transpose();

    // Camera centre via analytic determinant solution (homogeneous then dehomog)
    Eigen::Matrix3d Cmat;
    Cmat.col(0) = p2; Cmat.col(1) = p3; Cmat.col(2) = p4; double X = Cmat.determinant();
    Cmat.col(0) = p1; Cmat.col(1) = p3; Cmat.col(2) = p4; double Y = -Cmat.determinant();
    Cmat.col(0) = p1; Cmat.col(1) = p2; Cmat.col(2) = p4; double Z = Cmat.determinant();
    Cmat.col(0) = p1; Cmat.col(1) = p2; Cmat.col(2) = p3; double T = -Cmat.determinant();
    Eigen::Vector4d Pc_h(X, Y, Z, T);
    Pc_h /= Pc_h(3);
    Eigen::Vector3d Pc = Pc_h.head<3>();

    // Principal point
    Eigen::Vector3d pp_homogeneous = M * m3;
    pp_homogeneous /= pp_homogeneous(2);
    Eigen::Vector2d pp = pp_homogeneous.head<2>();

    // Principal vector
    Eigen::Vector3d pv = M.determinant() * m3;
    pv.normalize();

    // RQ decomposition -> K (upper triangular), Rc_w (rotation)
    auto [K, Rc_w] = rq3_impl(M);

    return {K, Rc_w, Pc, pp, pv};
}