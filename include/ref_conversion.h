#ifndef REF_CONVERSION_H
#define REF_CONVERSION_H

#include <Eigen/Dense>
#include <string>
#include "Calibration.h" // for TransformationData and utility IO

// Converts reference/world points to detector frame and returns:
// - Cmm_pts (Nx3) transformed world points
// - ref_dist_pts (4xN) homogeneous transformed dist points
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> ref_conversion(
    const std::string& position,
    const Eigen::MatrixXd& W,
    const Eigen::MatrixXd& distpts,
    const TransformationData& C2R,
    const TransformationData& C2D,
    const std::string& path_d);

#endif // REF_CONVERSION_H