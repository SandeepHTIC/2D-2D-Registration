#include "ref_conversion.h"
#include <filesystem>
#include <fstream>
#include <iomanip>

static void write_matrix(const std::string& filename, const Eigen::MatrixXd& matrix,
                         const std::string& delimiter = " ", int precision = 6) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    file << std::fixed << std::setprecision(precision);
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            file << matrix(i, j);
            if (j < matrix.cols() - 1) file << delimiter;
        }
        file << '\n';
    }
}

static Eigen::Matrix3d quat_to_rot(const std::vector<double>& q)
{
    // q: [w, x, y, z] with our input rearranged accordingly by caller
    double w = q[0], x = q[1], y = q[2], z = q[3];
    double n = std::sqrt(w*w + x*x + y*y + z*z);
    if (n > 0.0) { w/=n; x/=n; y/=n; z/=n; }
    Eigen::Matrix3d R;
    R << 1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
         2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
         2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y);
    return R;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> ref_conversion(
    const std::string& position,
    const Eigen::MatrixXd& W,
    const Eigen::MatrixXd& distpts,
    const TransformationData& C2R,
    const TransformationData& C2D,
    const std::string& path_d)
{
    // Reference marker transformation
    Eigen::Vector3d trans(C2R.tx, C2R.ty, C2R.tz);
    Eigen::Matrix3d rot = quat_to_rot({-C2R.rotation[3], C2R.rotation[0], C2R.rotation[1], C2R.rotation[2]});

    Eigen::Matrix4d cam2ref = Eigen::Matrix4d::Identity();
    cam2ref.block<3,3>(0,0) = rot;
    cam2ref.block<3,1>(0,3) = trans;

    // Detector marker transformation
    Eigen::Vector3d trans2(C2D.tx, C2D.ty, C2D.tz);
    Eigen::Matrix3d rot2 = quat_to_rot({-C2D.rotation[3], C2D.rotation[0], C2D.rotation[1], C2D.rotation[2]});

    Eigen::Matrix4d cam2DD = Eigen::Matrix4d::Identity();
    cam2DD.block<3,3>(0,0) = rot2;
    cam2DD.block<3,1>(0,3) = trans2;

    // Calculate transformations
    Eigen::Matrix4d ref2cam = cam2ref.inverse();
    Eigen::Matrix4d ref2DD = ref2cam * cam2DD;

    // Save ref2dd.txt for AP position
    if (position == "AP") {
        std::string ref2dd_dir = path_d + "\\Output\\" + position;
        std::filesystem::create_directories(ref2dd_dir);
        std::string ref2dd_file = ref2dd_dir + "\\ref2dd.txt";
        write_matrix(ref2dd_file, ref2DD, " ", 6);
    }

    // Transform world points
    Eigen::MatrixXd W_homogeneous(4, W.rows());
    W_homogeneous.block(0, 0, 3, W.rows()) = W.transpose();
    W_homogeneous.row(3) = Eigen::VectorXd::Ones(W.rows());

    Eigen::MatrixXd ref_DD = ref2DD * W_homogeneous;
    Eigen::MatrixXd Cmm_pts = ref_DD.block(0, 0, 3, W.rows()).transpose();

    // Transform distance points
    Eigen::MatrixXd distpts_homogeneous(4, distpts.rows());
    distpts_homogeneous.block(0, 0, 3, distpts.rows()) = distpts.transpose();
    distpts_homogeneous.row(3) = Eigen::VectorXd::Ones(distpts.rows());

    Eigen::MatrixXd ref_dist_pts = ref2DD * distpts_homogeneous;

    return {Cmm_pts, ref_dist_pts};
}