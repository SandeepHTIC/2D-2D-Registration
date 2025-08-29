#include "Calibration.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <numeric>
// Helpers split into separate translation units
#include "ref_conversion.h"
#include "estimate_camera_matrix.h"
#include "decompose_camera.h"
#include "registration_check.h"

CalibrationResult Calibration::calibrate(
    const std::string& position,
    const TransformationData& C2R,
    const TransformationData& C2D,
    const Eigen::MatrixXd& W,
    const Eigen::MatrixXd& Dpts,
    const std::string& path_d,
    int rewarp) {
    
    CalibrationResult result;
    result.success = false;
    result.RPE = 10.0;
    
    try {
        std::string folder = (rewarp == 1) ? "Rewarp\\" : "";
        
        Eigen::MatrixXd icpPlatFid = loadIcpPlatFid(path_d);
        
        Eigen::MatrixXd W_working = W;
        
        // Reference conversion via helper
        auto [Ref_CMM_pts, Ref_dist_pts] = ref_conversion(position, W_working, Dpts, C2R, C2D, path_d);
        
        std::string xy_file = path_d + "\\Output\\" + position + "\\PD\\" + position + "_2D.txt";
        Eigen::MatrixXd xy_o = readMatrix(xy_file);
        
        if (xy_o.rows() == 0) {
            throw std::runtime_error("Failed to read 2D points file: " + xy_file);
        }
        
        // Enforce strict label-based ordering to match world-point order
        // Validate labels are integers and 1-based within Ref_CMM range
        for (int i = 0; i < xy_o.rows(); ++i) {
            double lbl = xy_o(i, 2);
            if (std::floor(lbl) != lbl || lbl < 1 || lbl > Ref_CMM_pts.rows()) {
                std::ostringstream oss;
                oss << "Invalid label at row " << i << ": " << lbl
                    << " (expected integer in [1," << Ref_CMM_pts.rows() << "])";
                throw std::runtime_error(oss.str());
            }
        }

        // Build sortable rows: [x, y, label]
        struct Row { double x, y, label; };
        std::vector<Row> rows(xy_o.rows());
        for (int i = 0; i < xy_o.rows(); ++i) {
            rows[i] = {xy_o(i, 0), xy_o(i, 1), xy_o(i, 2)};
        }

        // Lexicographic sort: x desc, then y desc, then label desc
        std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
            if (a.x != b.x) return a.x > b.x;
            if (a.y != b.y) return a.y > b.y;
            return a.label > b.label;
        });

        // Rebuild xy_label_sorted and order (labels → 0-based)
        Eigen::MatrixXd xy_label_sorted(rows.size(), 3);
        Eigen::VectorXi order(rows.size());
        for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
            xy_label_sorted(i, 0) = rows[i].x;
            xy_label_sorted(i, 1) = rows[i].y;
            xy_label_sorted(i, 2) = rows[i].label;
            order(i) = static_cast<int>(rows[i].label) - 1;
        }

        
        // Get image height from the cropped image
        double r = getImageHeightFromCroppedImage(path_d);
        if (r <= 0) {
            // Fallback to blob image
            r = getImageHeightFromBlob(position, path_d);
            if (r <= 0) {
                // Final fallback to estimation from data
                r = getImageHeight(position, path_d, icpPlatFid);
                if (r <= 0) {
                    throw std::runtime_error("Could not determine image height from available sources");
                }
            }
        }
        
        // Extract xy coordinates and apply coordinate transformation
        Eigen::MatrixXd xy(xy_label_sorted.rows(), 2);
        xy.col(0) = xy_label_sorted.col(0);
        xy.col(1) = r - xy_label_sorted.col(1).array(); // Flip y-coordinate
        
        // Order Ref_CMM by labels
        Eigen::MatrixXd Ref_CMM_pts_order(order.size(), Ref_CMM_pts.cols());
        for (int i = 0; i < order.size(); ++i) {
            int lbl = order(i);
            if (lbl < 0 || lbl >= Ref_CMM_pts.rows()) {
                std::ostringstream oss;
                oss << "Label index out of bounds after sorting: " << lbl;
                throw std::runtime_error(oss.str());
            }
            Ref_CMM_pts_order.row(i) = Ref_CMM_pts.row(lbl);
        }

        // Debug: log image height and first few sorted 2D rows
        try {
            std::string dbg = path_d + "\\Output\\" + position + "\\debug_calib.txt";
            std::ofstream df(dbg, std::ios::app);
            if (df.is_open()) {
                df << "r=" << r << "\n";
                df << "xy_sorted_top5(x,y,label):\n";
                for (int i = 0; i < std::min<int>(xy_label_sorted.rows(), 5); ++i) {
                    df << std::fixed << std::setprecision(6)
                       << xy_label_sorted(i,0) << " " << xy_label_sorted(i,1) << " " << xy_label_sorted(i,2) << "\n";
                }
            }
        } catch(...) {}

        // Estimate camera matrix via helper
        auto [P0, reprojection_errors] = estimate_camera_matrix(xy, Ref_CMM_pts_order);
        if (!reprojection_errors.empty()) {
            double sum_err = 0.0, max_err = 0.0;
            for (double e : reprojection_errors) { sum_err += e; if (e > max_err) max_err = e; }
            double mean_err = sum_err / static_cast<double>(reprojection_errors.size());
            std::cout << "DLT reprojection error (px) - mean: " << mean_err << ", max: " << max_err << std::endl;
            try {
                std::string dbg = path_d + "\\Output\\" + position + "\\debug_calib.txt";
                std::ofstream df(dbg, std::ios::app);
                if (df.is_open()) {
                    df << "DLT mean_err=" << mean_err << " max_err=" << max_err << "\n";
                    df << "P0 (3x4):\n";
                    df << std::fixed << std::setprecision(6)
                       << P0(0,0) << " " << P0(0,1) << " " << P0(0,2) << " " << P0(0,3) << "\n"
                       << P0(1,0) << " " << P0(1,1) << " " << P0(1,2) << " " << P0(1,3) << "\n"
                       << P0(2,0) << " " << P0(2,1) << " " << P0(2,2) << " " << P0(2,3) << "\n";
                }
            } catch(...) {}
        }

        // Decompose camera matrix
        auto [K, R_ct, Pc1, pp1, pv1] = decompose_camera(P0);

        // Debug: dump xy (flipped) and first 10 ordered world points, and K/R/Pc1
        try {
            std::string dbg = path_d + "\\Output\\" + position + "\\debug_calib.txt";
            std::ofstream df(dbg, std::ios::app);
            if (df.is_open()) {
                df << "xy_top10_flipped(x,r-y):\n";
                for (int i = 0; i < std::min<int>(xy.rows(), 10); ++i) {
                    df << std::fixed << std::setprecision(6)
                       << xy(i,0) << " " << xy(i,1) << "\n";
                }
                df << "Ref_CMM_pts_order_top10(X Y Z):\n";
                for (int i = 0; i < std::min<int>(Ref_CMM_pts_order.rows(), 10); ++i) {
                    df << std::fixed << std::setprecision(6)
                       << Ref_CMM_pts_order(i,0) << " "
                       << Ref_CMM_pts_order(i,1) << " "
                       << Ref_CMM_pts_order(i,2) << "\n";
                }
                df << "K:\n";
                df << K(0,0) << " " << K(0,1) << " " << K(0,2) << "\n"
                   << K(1,0) << " " << K(1,1) << " " << K(1,2) << "\n"
                   << K(2,0) << " " << K(2,1) << " " << K(2,2) << "\n";
                df << "R_ct:\n";
                df << R_ct(0,0) << " " << R_ct(0,1) << " " << R_ct(0,2) << "\n"
                   << R_ct(1,0) << " " << R_ct(1,1) << " " << R_ct(1,2) << "\n"
                   << R_ct(2,0) << " " << R_ct(2,1) << " " << R_ct(2,2) << "\n";
                df << "Pc1: " << Pc1.transpose() << "\n";
            }
        } catch(...) {}
        
        Eigen::Matrix3d K_norm = K / K(2, 2);
        
        Eigen::Matrix<double, 3, 4> Rt3;
        Rt3.block<3, 3>(0, 0) = R_ct;
        Rt3.block<3, 1>(0, 3) = -R_ct * Pc1;
        
        Eigen::Matrix<double, 3, 4> P_norm = K_norm * Rt3;
        
        // Debug: dump P_norm
        try {
            std::string dbg = path_d + "\\Output\\" + position + "\\debug_calib.txt";
            std::ofstream df(dbg, std::ios::app);
            if (df.is_open()) {
                df << "P_norm (3x4):\n";
                df << std::fixed << std::setprecision(6)
                   << P_norm(0,0) << " " << P_norm(0,1) << " " << P_norm(0,2) << " " << P_norm(0,3) << "\n"
                   << P_norm(1,0) << " " << P_norm(1,1) << " " << P_norm(1,2) << " " << P_norm(1,3) << "\n"
                   << P_norm(2,0) << " " << P_norm(2,1) << " " << P_norm(2,2) << " " << P_norm(2,3) << "\n";
            }
        } catch(...) {}
        
        Eigen::VectorXd P_patient(12);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                P_patient(i * 4 + j) = P_norm(i, j);
            }
        }
        
        std::string p_file = path_d + "\\Output\\" + position + folder + "\\P_Imf" + position + ".txt";
        
        // Ensure directory exists
        std::string p_dir = path_d + "\\Output\\" + position + folder;
        std::filesystem::create_directories(p_dir);
        
        writeMatrix(p_file, P_patient.transpose(), " ", 6);

        // Debug: also dump P_patient on one line for quick diff
        try {
            std::string dbg_line = path_d + "\\Output\\" + position + folder + "\\P_Imf_line.txt";
            std::ofstream df(dbg_line);
            if (df.is_open()) {
                df << std::fixed << std::setprecision(6);
                for (int i = 0; i < 12; ++i) { if (i) df << ' '; df << P_patient(i); }
                df << "\n";
            }
        } catch(...) {}
        
        std::string main_p_file = path_d + "\\Output" + folder + "\\P_Imf.txt";
        
        // Ensure main output directory exists
        std::string main_dir = path_d + "\\Output" + folder;
        std::filesystem::create_directories(main_dir);
        
        if (position == "AP") {
            if (fileExists(main_p_file)) {
                Eigen::MatrixXd P2 = readMatrix(main_p_file);
                if (P2.rows() < 2) {
                    appendMatrix(main_p_file, P_patient.transpose(), " ", 6);
                }
            } else {
                writeMatrix(main_p_file, P_patient.transpose(), " ", 6);
            }
        } else if (position == "LP") {
            if (fileExists(main_p_file)) {
                Eigen::MatrixXd P2 = readMatrix(main_p_file);
                if (P2.rows() < 2) {
                    appendMatrix(main_p_file, P_patient.transpose(), " ", 6);
                }
            } else {
                writeMatrix(main_p_file, P_patient.transpose(), " ", 6);
            }
        }
        
        // Registration check
        double error = registration_check(P0, Ref_dist_pts, position, r, path_d, rewarp);
        
        result.Result_Matrix = Eigen::Matrix4d::Identity();
        result.Result_Matrix.block<3, 4>(0, 0) = P0;
        result.RPE = std::round(error * 1000.0) / 1000.0; // Round to 3 decimal places
        result.success = true;
        
        appLog("CALIBRATION", path_d);
        
    } catch (const std::exception& e) {
        result.success = false;
        result.RPE = 10.0;
        result.error_message = e.what();
        
        writeErrorLog("Calibration Failure: " + std::string(e.what()), path_d, position);
        
        std::cerr << "Calibration error: " << e.what() << std::endl;
    }
    
    return result;
}

double Calibration::getImageHeightFromCroppedImage(const std::string& path_d) {
    try {
        // Read the cropped image that maindcm.cpp saves
        // maindcm.cpp: cv::imwrite("cropped_output.png", cropped_input);
        std::string cropped_file = path_d + "\\cropped_output.png";
        cv::Mat cropped = cv::imread(cropped_file, cv::IMREAD_GRAYSCALE);
        
        if (!cropped.empty()) {
            std::cout << "Image height determined from cropped image: " << cropped.rows << std::endl;
            return static_cast<double>(cropped.rows);
        }
        
        return -1.0; // Indicate failure
        
    } catch (const std::exception& e) {
        std::cerr << "Error reading cropped image: " << e.what() << std::endl;
        return -1.0;
    }
}

double Calibration::getImageHeightFromBlob(const std::string& position, const std::string& path_d) {
    try {
        // Read blob image from EXACT path where maindcm.cpp saves it
        // maindcm.cpp: std::string bw_file = pd_dir + "\\" + position + "bw.png";
        // where pd_dir = output_dir + "\\Output\\" + position + "\\PD"
        std::string blob_file = path_d + "\\Output\\" + position + "\\PD\\" + position + "bw.png";
        cv::Mat blob = cv::imread(blob_file, cv::IMREAD_GRAYSCALE);
        
        if (!blob.empty()) {
            std::cout << "Image height determined from blob image: " << blob.rows << std::endl;
            return static_cast<double>(blob.rows);
        }
        
        return -1.0; // Indicate failure
        
    } catch (const std::exception& e) {
        std::cerr << "Error reading blob image: " << e.what() << std::endl;
        return -1.0;
    }
}

double Calibration::getImageHeight(const std::string& position, const std::string& path_d, const Eigen::MatrixXd& icpPlatFid) {
    try {
        // Method 1: Use icpPlatFid data to estimate image bounds
        if (icpPlatFid.rows() > 0 && icpPlatFid.cols() >= 2) {
            double max_y = icpPlatFid.col(1).maxCoeff();
            double min_y = icpPlatFid.col(1).minCoeff();
            double estimated_height = max_y + (max_y - min_y) * 0.1; // Add 10% margin
            
            std::cout << "Estimated image height from icpPlatFid data: " << estimated_height << std::endl;
            std::cout << "  - Y range: " << min_y << " to " << max_y << std::endl;
            
            // Validate the estimation (should be reasonable image size)
            if (estimated_height > 100 && estimated_height < 10000) {
                return estimated_height;
            }
        }
        
        // Method 2: Try to read the original cropped image from EXACT path where maindcm.cpp saves it
        // maindcm.cpp: cv::imwrite("cropped_output.png", cropped_input);
        std::string cropped_file = path_d + "\\cropped_output.png";
        cv::Mat cropped = cv::imread(cropped_file, cv::IMREAD_GRAYSCALE);
        
        if (!cropped.empty()) {
            std::cout << "Image height determined from cropped image: " << cropped.rows << std::endl;
            return static_cast<double>(cropped.rows);
        }
        
        // Method 3: Use 2D points file to estimate bounds from EXACT path where maindcm.cpp saves it
        // maindcm.cpp: std::string xy_file = pd_dir + "\\" + position + "_2D.txt";
        std::string xy_file = path_d + "\\Output\\" + position + "\\PD\\" + position + "_2D.txt";
        Eigen::MatrixXd xy_data = readMatrix(xy_file);
        if (xy_data.rows() > 0 && xy_data.cols() >= 2) {
            double max_y_2d = xy_data.col(1).maxCoeff();
            double min_y_2d = xy_data.col(1).minCoeff();
            double estimated_height_2d = max_y_2d + (max_y_2d - min_y_2d) * 0.1;
            
            std::cout << "Estimated image height from 2D points: " << estimated_height_2d << std::endl;
            
            if (estimated_height_2d > 100 && estimated_height_2d < 10000) {
                return estimated_height_2d;
            }
        }
        
        std::cerr << "Warning: Could not auto-determine image height. Using default value of 1024." << std::endl;
        return 1024.0; // Default fallback value
        
    } catch (const std::exception& e) {
        std::cerr << "Error determining image height: " << e.what() << std::endl;
        return 1024.0; // Default fallback value
    }
}

// moved to src/ref_conversion.cpp

// moved to src/estimate_camera_matrix.cpp

// moved to src/decompose_camera.cpp

// moved to src/registration_check.cpp
        



// Utility function implementations
Eigen::Matrix3d Calibration::quaternionToRotationMatrix(const std::vector<double>& q) {
    // Normalize quaternion
    double w = q[0], x = q[1], y = q[2], z = q[3];
    double norm = std::sqrt(w*w + x*x + y*y + z*z);
    if (norm > 0.0) {
        w /= norm; x /= norm; y /= norm; z /= norm;
    }

    Eigen::Matrix3d R;
    R << 1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
         2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
         2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y);

    return R;
}

std::pair<Eigen::MatrixXd, Eigen::Matrix3d> Calibration::normalise2dpts(const Eigen::MatrixXd& pts) {
    if (pts.rows() != 3) {
        throw std::invalid_argument("pts must be 3xN");
    }
    
    Eigen::MatrixXd newpts = pts;
    
    // Find finite points
    std::vector<int> finiteind;
    for (int i = 0; i < pts.cols(); ++i) {
        if (std::abs(pts(2, i)) > std::numeric_limits<double>::epsilon()) {
            finiteind.push_back(i);
        }
    }
    
    // Normalize finite points
    for (int idx : finiteind) {
        newpts(0, idx) /= newpts(2, idx);
        newpts(1, idx) /= newpts(2, idx);
        newpts(2, idx) = 1.0;
    }
    
    // Calculate centroid
    Eigen::Vector2d c = Eigen::Vector2d::Zero();
    for (int idx : finiteind) {
        c(0) += newpts(0, idx);
        c(1) += newpts(1, idx);
    }
    c /= finiteind.size();
    
    // Shift origin to centroid
    for (int idx : finiteind) {
        newpts(0, idx) -= c(0);
        newpts(1, idx) -= c(1);
    }
    
    // Calculate mean distance
    double meandist = 0.0;
    for (int idx : finiteind) {
        meandist += std::sqrt(newpts(0, idx)*newpts(0, idx) + newpts(1, idx)*newpts(1, idx));
    }
    meandist /= finiteind.size();
    
    double scale = std::sqrt(2.0) / meandist;
    
    // Create transformation matrix
    Eigen::Matrix3d T;
    T << scale, 0, -scale*c(0),
         0, scale, -scale*c(1),
         0, 0, 1;
    
    return {T * pts, T};
}

Eigen::Matrix<double, 3, 4> Calibration::refineProjectionMatrix(
    const Eigen::Matrix<double, 3, 4>& P_init,
    const Eigen::MatrixXd& imagePoints_homogeneous,
    const Eigen::MatrixXd& worldPoints) {
    
    // Simple iterative refinement using Gauss-Newton method
    Eigen::Matrix<double, 3, 4> P = P_init;
    const int max_iterations = 5; // Reduced iterations for stability
    const double tolerance = 1e-8;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Calculate current reprojection errors
        Eigen::MatrixXd worldPoints_homogeneous(worldPoints.rows(), 4);
        worldPoints_homogeneous.block(0, 0, worldPoints.rows(), 3) = worldPoints;
        worldPoints_homogeneous.col(3) = Eigen::VectorXd::Ones(worldPoints.rows());
        
        double total_error = 0.0;
        Eigen::MatrixXd J(2 * worldPoints.rows(), 12); // Jacobian matrix
        Eigen::VectorXd residuals(2 * worldPoints.rows());
        
        for (int i = 0; i < worldPoints.rows(); ++i) {
            Eigen::Vector4d X = worldPoints_homogeneous.row(i);
            Eigen::Vector3d x_proj = P * X;
            
            if (std::abs(x_proj(2)) < 1e-10) continue; // Skip degenerate points
            
            double u = x_proj(0) / x_proj(2);
            double v = x_proj(1) / x_proj(2);
            
            // Observed image coordinates
            double u_obs = imagePoints_homogeneous(0, i);
            double v_obs = imagePoints_homogeneous(1, i);
            
            // Residuals
            residuals(2*i) = u_obs - u;
            residuals(2*i+1) = v_obs - v;
            
            total_error += residuals(2*i)*residuals(2*i) + residuals(2*i+1)*residuals(2*i+1);
            
            // Compute Jacobian (derivatives of projection w.r.t. P matrix elements)
            double w = x_proj(2);
            double w2 = w * w;
            
            // du/dP and dv/dP
            for (int j = 0; j < 4; ++j) {
                // First row of P (affects u)
                J(2*i, j) = X(j) / w;
                J(2*i, 4+j) = 0;
                J(2*i, 8+j) = -u * X(j) / w;
                
                // Second row of P (affects v)
                J(2*i+1, j) = 0;
                J(2*i+1, 4+j) = X(j) / w;
                J(2*i+1, 8+j) = -v * X(j) / w;
            }
        }
        
        // Solve normal equations: (J^T J) delta = J^T residuals
        Eigen::MatrixXd JtJ = J.transpose() * J;
        Eigen::VectorXd Jtr = J.transpose() * residuals;
        
        // Add damping for numerical stability (Levenberg-Marquardt style)
        double lambda = 1e-6;
        JtJ.diagonal().array() += lambda;
        
        Eigen::VectorXd delta = JtJ.ldlt().solve(Jtr);
        
        // Update P matrix
        Eigen::Matrix<double, 3, 4> P_new = P;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                P_new(i, j) += delta(i*4 + j);
            }
        }
        
        // Check convergence
        double delta_norm = delta.norm();
        if (delta_norm < tolerance) {
            P = P_new;
            break;
        }
        
        P = P_new;
    }
    
    return P;
}

std::pair<Eigen::Matrix3d, Eigen::Matrix3d> Calibration::rq3(const Eigen::Matrix3d& A_in) {
    Eigen::Matrix3d A = A_in;
    double eps = 1e-10;
    // Step 1: Qx to zero out A(2,1) (MATLAB A(2,1) == C++ A(1,0))
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

Eigen::MatrixXd Calibration::readMatrix(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return Eigen::MatrixXd(0, 0);
    }
    
    std::vector<std::vector<double>> data;
    std::string line;
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<double> row;
        double value;
        
        while (iss >> value) {
            row.push_back(value);
        }
        
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    
    if (data.empty()) {
        return Eigen::MatrixXd(0, 0);
    }
    
    int rows = data.size();
    int cols = data[0].size();
    
    Eigen::MatrixXd matrix(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix(i, j) = data[i][j];
        }
    }
    
    return matrix;
}

void Calibration::writeMatrix(const std::string& filename, const Eigen::MatrixXd& matrix, 
                             const std::string& delimiter, int precision) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    file << std::fixed << std::setprecision(precision);
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            file << matrix(i, j);
            if (j < matrix.cols() - 1) {
                file << delimiter;
            }
        }
        file << std::endl;
    }
}

void Calibration::appendMatrix(const std::string& filename, const Eigen::MatrixXd& matrix, 
                              const std::string& delimiter, int precision) {
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for appending: " + filename);
    }
    
    file << std::fixed << std::setprecision(precision);
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            file << matrix(i, j);
            if (j < matrix.cols() - 1) {
                file << delimiter;
            }
        }
        file << std::endl;
    }
}

bool Calibration::fileExists(const std::string& filename) {
    return std::filesystem::exists(filename);
}



void Calibration::writeErrorLog(const std::string& message, const std::string& path_d, 
                               const std::string& position) {
    try {
        std::string outPath = path_d + "\\Output\\" + position;
        std::filesystem::create_directories(outPath);
        std::string fullFileName = outPath + "\\errorLogFile.txt";
        
        std::ofstream file(fullFileName);
        if (file.is_open()) {
            file << message << std::endl;
            file.close();
        }
        
        // Write to general error log
        std::string generalErrorFile = path_d + "\\errorLogFile.txt";
        std::ofstream generalFile(generalErrorFile, std::ios::app);
        if (generalFile.is_open()) {
            generalFile << message << std::endl;
            generalFile.close();
        }
        
        // Write to error store file
        std::string errorStoreFile = outPath + "\\errorStoreFile.txt";
        std::ofstream storeFile(errorStoreFile, std::ios::app);
        if (storeFile.is_open()) {
            storeFile << "Calibration Failure" << std::endl;
            storeFile << message << std::endl;
            storeFile.close();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error writing log files: " << e.what() << std::endl;
    }
}

void Calibration::appLog(const std::string& operation, const std::string& path_d) {
    try {
        std::string logFile = path_d + "\\appLog.txt";
        std::ofstream file(logFile, std::ios::app);
        if (file.is_open()) {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            file << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") 
                 << " - Operation completed: " << operation << " in " << path_d << std::endl;
            file.close();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error writing app log: " << e.what() << std::endl;
    }
    
    std::cout << "Operation completed: " << operation << " in " << path_d << std::endl;
}

Eigen::MatrixXd Calibration::loadIcpPlatFid(const std::string& path_d) {
    // Try to load ICP fiducial data from EXACT path where maindcm.cpp saves it
    // maindcm.cpp: std::string icpPlatFid_file = output_dir + "\\icpPlatFid.txt";
    std::vector<std::string> possiblePaths = {
        path_d + "\\icpPlatFid.txt",               // PRIMARY: Exact location where maindcm saves it
        path_d + "\\icp_2D.txt",                    // Alternative location
        path_d + "\\Output\\icpPlatFid.txt",       // In output directory
        "icpPlatFid.txt",                          // Current directory fallback
        "icp_2D.txt"                               // Current directory fallback
    };
    
    for (const std::string& icpFile : possiblePaths) {
        if (fileExists(icpFile)) {
            std::cout << "Loading ICP fiducial data from: " << icpFile << std::endl;
            Eigen::MatrixXd data = readMatrix(icpFile);
            if (data.rows() > 0) {
                std::cout << "Successfully loaded " << data.rows() << " ICP fiducial points." << std::endl;
                return data;
            }
        }
    }
    
    std::cout << "Warning: No ICP fiducial data found. Checked paths:" << std::endl;
    for (const std::string& path : possiblePaths) {
        std::cout << "  - " << path << std::endl;
    }
    
    // Return empty matrix if no data available
    return Eigen::MatrixXd(0, 0);
}