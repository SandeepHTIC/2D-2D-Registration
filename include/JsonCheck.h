#ifndef JSON_CHECK_H
#define JSON_CHECK_H

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <json.hpp>

struct JsonCheckResult {
    // Parsed data
    std::string position;              // AP or LP
    std::string imagePath;             // Input image path
    std::string registrationType;      // Expected: 2D2D
    std::vector<double> cropRoi;       // 4 elements
    Eigen::MatrixXd CMM_WorldPoints;   // Nx3
    Eigen::MatrixXd CMM_Dist_pts;      // Mx3

    // Paths
    std::string outputJsonPath;        // Output\\output<position>.json

    // Status
    bool ok;
    std::string errorMessage;
};


JsonCheckResult json_check_cpp(const nlohmann::json& input_SSR, const std::string& otpath, const std::string& path_d);

#endif // JSON_CHECK_H