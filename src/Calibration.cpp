#include "Calibration.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <fstream>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <filesystem>
#include <chrono>
#include <ctime>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "rq3.h"
#include "normalise2dpts.h"
#include "decompose_camera.h"
#include "RegistrationCheck.h"

CalibrationResult Calibration::calibrate(
    const std::string &position,
    const TransformationData &C2R,
    const TransformationData &C2D,
    const Eigen::MatrixXd &W,
    const Eigen::MatrixXd &Dpts,
    const std::string &path_d,
    int rewarp)
{

    CalibrationResult result;
    result.success = false;
    result.RPE = 10.0;

    try
    {

        std::string folder = (rewarp == 1) ? std::string("\\Rewarp") : std::string("");

        Eigen::MatrixXd icpPlatFid = loadIcpPlatFid(path_d);

        Eigen::MatrixXd W_working = W;

        // Reference conversion
        auto [Ref_CMM_pts, Ref_dist_pts] = refConversion(position, W_working, Dpts, C2R, C2D, path_d);

        std::string xy_file = path_d + "\\Output\\" + position + "\\PD\\" + position + "_2D.txt";

        Eigen::MatrixXd xy_o = readMatrix(xy_file);

        if (xy_o.rows() == 0)
        {
            throw std::runtime_error("Failed to read 2D points file: " + xy_file);
        }

        if (xy_o.cols() < 3)
        {
            throw std::runtime_error("2D points file must have at least 3 columns: x y label");
        }
        const int max_label = static_cast<int>(Ref_CMM_pts.rows());
        std::unordered_map<int, Eigen::Vector2d> xy_by_label;
        xy_by_label.reserve(xy_o.rows());
        for (int i = 0; i < xy_o.rows(); ++i)
        {
            double lbl_raw = xy_o(i, 2);
            int lbl = static_cast<int>(std::llround(lbl_raw));
            if (std::abs(lbl_raw - static_cast<double>(lbl)) > 1e-9)
            {
                std::cerr << "Skipping non-integer label at row " << i << ": " << lbl_raw << "\n";
                continue;
            }
            if (lbl < 1 || lbl > max_label)
            {
                std::cerr << "Skipping out-of-range label at row " << i << ": " << lbl << "\n";
                continue;
            }
            if (xy_by_label.find(lbl) != xy_by_label.end())
            {
                // Duplicate label: keep the first, skip others
                std::cerr << "Duplicate label " << lbl << ", keeping the first occurrence\n";
                continue;
            }
            xy_by_label[lbl] = Eigen::Vector2d(xy_o(i, 0), xy_o(i, 1));
        }

        // Prefer PD blob image height to match the image used for 2D detection
        double r = getImageHeightFromBlob(position, path_d);
        if (r <= 0)
        {
            // Fallback to cropped image
            r = getImageHeightFromCroppedImage(path_d);
            if (r <= 0)
            {
                // Final fallback to estimation from data
                r = getImageHeight(position, path_d, icpPlatFid);
                if (r <= 0)
                {
                    throw std::runtime_error("Could not determine image height from available sources");
                }
            }
        }

        // Build matched lists by ascending label order to align with Ref_CMM rows
        std::vector<int> labels;
        labels.reserve(xy_by_label.size());
        for (int l = 1; l <= max_label; ++l)
        {
            if (xy_by_label.find(l) != xy_by_label.end())
                labels.push_back(l);
        }
        if (static_cast<int>(labels.size()) < 8)
        {
            throw std::runtime_error("Insufficient matched labeled points (need >= 8) to estimate camera");
        }

        Eigen::MatrixXd xy(labels.size(), 2);
        Eigen::MatrixXd Ref_CMM_pts_order(labels.size(), Ref_CMM_pts.cols());
        for (size_t i = 0; i < labels.size(); ++i)
        {
            int lbl = labels[i];
            const auto &pt = xy_by_label[lbl];
            xy(i, 0) = pt.x();
            xy(i, 1) = r - pt.y(); // Flip y-coordinate
            Ref_CMM_pts_order.row(static_cast<int>(i)) = Ref_CMM_pts.row(lbl - 1);
        }

        // RANSAC over minimal subsets (6 points) for robust DLT
        const int N = static_cast<int>(labels.size());
        const int s = 6;
        const int max_iters = std::min(5000, 20 * N);
        const double thresh_px = 0.25;
        std::mt19937 rng{std::random_device{}()};

        // Prepack world points and image points
        std::vector<Eigen::Vector3d> Xw(N);
        std::vector<Eigen::Vector2d> Uv(N);
        for (int i = 0; i < N; ++i)
        {
            Xw[i] = Ref_CMM_pts_order.row(i).transpose();
            Uv[i] = xy.row(i).transpose();
        }

        // Estimate projection using all points, then refine
        Eigen::Matrix<double, 3, 4> P0;
        {
            auto [P0_all, _] = estimateCameraMatrix(xy, Ref_CMM_pts_order);
            Eigen::MatrixXd img_h(3, xy.rows());
            img_h.block(0, 0, 2, xy.rows()) = xy.transpose();
            img_h.row(2) = Eigen::VectorXd::Ones(xy.rows());
            P0 = refineProjectionMatrix(P0_all, img_h, Ref_CMM_pts_order);
        }

        // Compute reprojection errors for reporting
        std::vector<double> reprojection_errors;
        reprojection_errors.reserve(N);
        for (int i = 0; i < N; ++i)
        {
            Eigen::Vector4d Xh(Xw[i].x(), Xw[i].y(), Xw[i].z(), 1.0);
            Eigen::Vector3d p = P0 * Xh;
            double w = p(2);
            if (std::abs(w) < 1e-15)
            {
                reprojection_errors.push_back(1e6);
                continue;
            }
            double du = Uv[i].x() - p(0) / w;
            double dv = Uv[i].y() - p(1) / w;
            reprojection_errors.push_back(std::sqrt(du * du + dv * dv));
        }

        if (!reprojection_errors.empty())
        {
            double sum_err = 0.0, max_err = 0.0;
            for (double e : reprojection_errors)
            {
                sum_err += e;
                if (e > max_err)
                    max_err = e;
            }
            double mean_err = sum_err / static_cast<double>(reprojection_errors.size());

            double final_mean = mean_err;
            double final_max = max_err;

            // Optional non-linear refinement to reduce reprojection error further
            if (mean_err > 0.3)
            {
                Eigen::MatrixXd img_h(3, xy.rows());
                img_h.block(0, 0, 2, xy.rows()) = xy.transpose();
                img_h.row(2) = Eigen::VectorXd::Ones(xy.rows());
                P0 = refineProjectionMatrix(P0, img_h, Ref_CMM_pts_order);
                // Recompute reprojection error after refinement
                Eigen::MatrixXd X_h(4, Ref_CMM_pts_order.rows());
                X_h.block(0, 0, 3, Ref_CMM_pts_order.rows()) = Ref_CMM_pts_order.transpose();
                X_h.row(3) = Eigen::VectorXd::Ones(Ref_CMM_pts_order.rows());
                double sum2 = 0.0, max2 = 0.0;
                for (int i = 0; i < Ref_CMM_pts_order.rows(); ++i)
                {
                    Eigen::Vector3d proj = P0 * X_h.col(i);
                    double u = proj(0) / proj(2);
                    double v = proj(1) / proj(2);
                    double dx = u - xy(i, 0);
                    double dy = v - xy(i, 1);
                    double e = std::sqrt(dx * dx + dy * dy);
                    sum2 += e;
                    if (e > max2)
                        max2 = e;
                }
                final_mean = sum2 / static_cast<double>(Ref_CMM_pts_order.rows());
                final_max = max2;
            }
        }

        // Decompose camera matrix
        auto [K, R_ct, Pc1, pp1, pv1] = decomposeCamera(P0);
        // Normalize K to have K(2,2)=1 for consistency with MATLAB
        if (std::abs(K(2, 2)) > 1e-15)
        {
            K /= K(2, 2);
        }

        Eigen::Matrix3d K_norm = K;

        // Create Rt3 matrix
        Eigen::Matrix<double, 3, 4> Rt3;
        Rt3.block<3, 3>(0, 0) = R_ct;
        Rt3.block<3, 1>(0, 3) = -R_ct * Pc1;

        // Calculate normalized projection matrix (matching MATLAB: P_norm = K_norm*Rt3)
        Eigen::Matrix<double, 3, 4> P_norm = K_norm * Rt3;

        // Reshape to 1x12 vector (matching MATLAB: P_patient=reshape(P_norm',[1,12]))
        Eigen::VectorXd P_patient(12);
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                P_patient(i * 4 + j) = P_norm(i, j);
            }
        }

        std::string p_file = path_d + "\\Output\\" + position + folder + "\\P_Imf" + position + ".txt";

        // Ensure directory exists
        std::string p_dir = path_d + "\\Output\\" + position + folder;
        std::filesystem::create_directories(p_dir);

        writeMatrix(p_file, P_patient.transpose(), " ", 6);

        // Write/append to common P_Imf.txt (append when file exists)
        std::string main_p_file = path_d + "\\Output" + folder + "\\P_Imf.txt";
        std::filesystem::create_directories(path_d + "\\Output" + folder);
        if (fileExists(main_p_file))
        {
            appendMatrix(main_p_file, P_patient.transpose(), " ", 6);
        }
        else
        {
            writeMatrix(main_p_file, P_patient.transpose(), " ", 6);
        }

        // Registration check (matching MATLAB: Error=Registration_check(P0',Ref_dist_pts,position,r,path_d,rewarp))
        double error = registrationCheck(P0, Ref_dist_pts, position, r, path_d, rewarp);

        // Prepare result (matching MATLAB: js.Result_Matrix=[P0';0 0 0 1]; js.RPE = round(Error,3))
        result.Result_Matrix = Eigen::Matrix4d::Identity();
        result.Result_Matrix.block<3, 4>(0, 0) = P0;
        result.RPE = std::round(error * 1000.0) / 1000.0; // Round to 3 decimal places
        result.success = true;

        // Log success (matching MATLAB: appLog('CALIBRATION',path_d))
        appLog("CALIBRATION", path_d);
    }
    catch (const std::exception &e)
    {
        // Error handling matching MATLAB catch block
        result.success = false;
        result.RPE = 10.0;
        result.error_message = e.what();

        // Write error logs (matching MATLAB error logging)
        writeErrorLog("Calibration Failure: " + std::string(e.what()), path_d, position);

        std::cerr << "Calibration error: " << e.what() << std::endl;
    }

    return result;
}

double Calibration::getImageHeightFromCroppedImage(const std::string &path_d)
{
    try
    {
        // Read the cropped image that maindcm.cpp saves
        // maindcm.cpp: cv::imwrite("cropped_output.png", cropped_input);
        std::string cropped_file = path_d + "\\cropped_output.png";
        cv::Mat cropped = cv::imread(cropped_file, cv::IMREAD_GRAYSCALE);
        if (!cropped.empty())
        {
            return static_cast<double>(cropped.rows);
        }
        return -1.0; // Indicate failure
    }
    catch (const std::exception &)
    {
        return -1.0;
    }
}

double Calibration::getImageHeightFromBlob(const std::string &position, const std::string &path_d)
{
    try
    {
        std::string blob_file = path_d + "\\Output\\" + position + "\\PD\\" + position + "bw.png";
        cv::Mat blob = cv::imread(blob_file, cv::IMREAD_GRAYSCALE);
        if (!blob.empty())
        {
            return static_cast<double>(blob.rows);
        }
        return -1.0; // Indicate failure
    }
    catch (const std::exception &)
    {
        return -1.0;
    }
}

double Calibration::getImageHeight(const std::string &position, const std::string &path_d, const Eigen::MatrixXd &icpPlatFid)
{
    try
    {

        if (icpPlatFid.rows() > 0 && icpPlatFid.cols() >= 2)
        {
            double max_y = icpPlatFid.col(1).maxCoeff();
            double min_y = icpPlatFid.col(1).minCoeff();
            double estimated_height = max_y + (max_y - min_y) * 0.05; // Add 5% margin (stricter)
            if (estimated_height > 100 && estimated_height < 10000)
            {
                return estimated_height;
            }
        }

        std::string cropped_file = path_d + "\\cropped_output.png";
        cv::Mat cropped = cv::imread(cropped_file, cv::IMREAD_GRAYSCALE);
        if (!cropped.empty())
        {
            return static_cast<double>(cropped.rows);
        }

        std::string xy_file = path_d + "\\Output\\" + position + "\\PD\\" + position + "_2D.txt";
        Eigen::MatrixXd xy_data = readMatrix(xy_file);
        if (xy_data.rows() > 0 && xy_data.cols() >= 2)
        {
            double max_y_2d = xy_data.col(1).maxCoeff();
            double min_y_2d = xy_data.col(1).minCoeff();
            double estimated_height_2d = max_y_2d + (max_y_2d - min_y_2d) * 0.05;
            if (estimated_height_2d > 100 && estimated_height_2d < 10000)
            {
                return estimated_height_2d;
            }
        }
        return 1024.0; // Default fallback value
    }
    catch (const std::exception &)
    {
        return 1024.0; // Default fallback value
    }
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> Calibration::refConversion(
    const std::string &position,
    const Eigen::MatrixXd &W,
    const Eigen::MatrixXd &distpts,
    const TransformationData &C2R,
    const TransformationData &C2D,
    const std::string &path_d)
{

    // Reference marker transformation
    Eigen::Vector3d trans(C2R.tx, C2R.ty, C2R.tz);
    // JSON stores quaternion as [x, y, z, w]; MATLAB uses [-w, x, y, z]
    Eigen::Matrix3d rot = quaternionToRotationMatrix({-C2R.rotation[3], C2R.rotation[0], C2R.rotation[1], C2R.rotation[2]});

    Eigen::Matrix4d cam2ref = Eigen::Matrix4d::Identity();
    cam2ref.block<3, 3>(0, 0) = rot;
    cam2ref.block<3, 1>(0, 3) = trans;

    // Detector marker transformation
    Eigen::Vector3d trans2(C2D.tx, C2D.ty, C2D.tz);
    // JSON stores quaternion as [x, y, z, w]; MATLAB uses [-w, x, y, z]
    Eigen::Matrix3d rot2 = quaternionToRotationMatrix({-C2D.rotation[3], C2D.rotation[0], C2D.rotation[1], C2D.rotation[2]});

    Eigen::Matrix4d cam2DD = Eigen::Matrix4d::Identity();
    cam2DD.block<3, 3>(0, 0) = rot2;
    cam2DD.block<3, 1>(0, 3) = trans2;

    // Calculate transformations
    Eigen::Matrix4d ref2cam = cam2ref.inverse();
    Eigen::Matrix4d ref2DD = ref2cam * cam2DD;

    // Save ref2dd.txt for AP position
    if (position == "AP")
    {
        std::string ref2dd_dir = path_d + "\\Output\\" + position;
        std::filesystem::create_directories(ref2dd_dir);
        std::string ref2dd_file = ref2dd_dir + "\\ref2dd.txt";
        writeMatrix(ref2dd_file, ref2DD, " ", 6);
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

std::pair<Eigen::Matrix<double, 3, 4>, std::vector<double>> Calibration::estimateCameraMatrix(
    const Eigen::MatrixXd &imagePoints,
    const Eigen::MatrixXd &worldPoints)
{

    const int M = static_cast<int>(worldPoints.rows());
    if (M < 6 || imagePoints.rows() != M || imagePoints.cols() != 2 || worldPoints.cols() != 3)
    {
        return {Eigen::Matrix<double, 3, 4>::Zero(), std::vector<double>()};
    }

    Eigen::MatrixXd imagePoints_h(3, M);
    imagePoints_h.block(0, 0, 2, M) = imagePoints.transpose();
    imagePoints_h.row(2) = Eigen::VectorXd::Ones(M);
    auto [pts_norm, T] = normalise2dpts(imagePoints_h);
    Eigen::Matrix3d Tinv = T.inverse();

    Eigen::MatrixXd A(2 * M, 12);
    for (int i = 0; i < M; ++i)
    {
        double X = worldPoints(i, 0), Y = worldPoints(i, 1), Z = worldPoints(i, 2);
        double u = pts_norm(0, i), v = pts_norm(1, i);
        A.row(2 * i) << X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u;
        A.row(2 * i + 1) << 0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v;
    }

    Eigen::MatrixXd AtA = A.transpose() * A;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(AtA);
    Eigen::Index minIndex;
    es.eigenvalues().minCoeff(&minIndex);
    Eigen::VectorXd P_vec = es.eigenvectors().col(minIndex);

    Eigen::Matrix<double, 4, 3> camMatrix_prime;
    camMatrix_prime << P_vec(0), P_vec(4), P_vec(8),
        P_vec(1), P_vec(5), P_vec(9),
        P_vec(2), P_vec(6), P_vec(10),
        P_vec(3), P_vec(7), P_vec(11);
    camMatrix_prime = camMatrix_prime * Tinv.transpose();
    if (camMatrix_prime(3, 2) < 0)
        camMatrix_prime = -camMatrix_prime;

    Eigen::Matrix<double, 3, 4> P = camMatrix_prime.transpose();

    std::vector<double> reprojectionErrors;
    reprojectionErrors.reserve(M);
    for (int i = 0; i < M; ++i)
    {
        Eigen::Vector4d X(worldPoints(i, 0), worldPoints(i, 1), worldPoints(i, 2), 1.0);
        Eigen::Vector3d p = P * X;
        double w = p(2);
        if (std::abs(w) < 1e-12)
        {
            reprojectionErrors.push_back(1e6);
            continue;
        }
        double uu = p(0) / w, vv = p(1) / w;
        double du = imagePoints(i, 0) - uu;
        double dv = imagePoints(i, 1) - vv;
        reprojectionErrors.push_back(std::sqrt(du * du + dv * dv));
    }

    return {P, reprojectionErrors};
}

std::tuple<Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector2d, Eigen::Vector3d>
Calibration::decomposeCamera(const Eigen::Matrix<double, 3, 4> &P)
{
    // Delegate to header-only helper for clarity and testability
    return decompose_camera(P);
}

Eigen::Matrix3d Calibration::quaternionToRotationMatrix(const std::vector<double> &q)
{
    if (q.size() != 4)
        throw std::invalid_argument("Quaternion must have 4 elements [w,x,y,z]");
    double w = q[0], x = q[1], y = q[2], z = q[3];
    double n = std::sqrt(w * w + x * x + y * y + z * z);
    if (n < 1e-12)
        throw std::invalid_argument("Quaternion norm is zero");
    w /= n;
    x /= n;
    y /= n;
    z /= n;
    Eigen::Matrix3d R;
    R << 1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y);
    return R;
}

// Adapter to header normalisation function
std::pair<Eigen::MatrixXd, Eigen::Matrix3d> Calibration::normalise2dpts(const Eigen::MatrixXd &pts)
{
    return ::normalise2dpts(pts);
}

// File utilities
bool Calibration::fileExists(const std::string &filename)
{
    return std::filesystem::exists(filename);
}

Eigen::MatrixXd Calibration::readMatrix(const std::string &filename)
{
    std::ifstream ifs(filename);
    if (!ifs.is_open())
        return Eigen::MatrixXd();
    std::vector<std::vector<double>> rows;
    std::string line;
    size_t cols = 0;
    while (std::getline(ifs, line))
    {
        std::stringstream ss(line);
        std::vector<double> values;
        double v;
        while (ss >> v)
            values.push_back(v);
        if (!values.empty())
        {
            if (cols == 0)
                cols = values.size();
            rows.push_back(std::move(values));
        }
    }
    if (rows.empty() || cols == 0)
        return Eigen::MatrixXd();
    Eigen::MatrixXd M(rows.size(), cols);
    for (size_t i = 0; i < rows.size(); ++i)
        for (size_t j = 0; j < cols; ++j)
            M(i, j) = rows[i][j];
    return M;
}

void Calibration::writeMatrix(const std::string &filename, const Eigen::MatrixXd &matrix,
                              const std::string &delimiter, int precision)
{
    std::filesystem::create_directories(std::filesystem::path(filename).parent_path());
    std::ofstream ofs(filename);
    ofs << std::fixed << std::setprecision(precision);
    for (int i = 0; i < matrix.rows(); ++i)
    {
        for (int j = 0; j < matrix.cols(); ++j)
        {
            ofs << matrix(i, j);
            if (j + 1 < matrix.cols())
                ofs << delimiter;
        }
        ofs << '\n';
    }
}

void Calibration::appendMatrix(const std::string &filename, const Eigen::MatrixXd &matrix,
                               const std::string &delimiter, int precision)
{
    std::filesystem::create_directories(std::filesystem::path(filename).parent_path());
    std::ofstream ofs(filename, std::ios::app);
    ofs << std::fixed << std::setprecision(precision);
    for (int i = 0; i < matrix.rows(); ++i)
    {
        for (int j = 0; j < matrix.cols(); ++j)
        {
            ofs << matrix(i, j);
            if (j + 1 < matrix.cols())
                ofs << delimiter;
        }
        ofs << '\n';
    }
}

void Calibration::writeErrorLog(const std::string &message, const std::string &path_d,
                                const std::string &position)
{
    try
    {
        std::string outPath = path_d + "\\Output\\" + position;
        std::filesystem::create_directories(outPath);
        // errorLogFile.txt (overwrite)
        {
            std::ofstream fid(outPath + "\\errorLogFile.txt");
            fid << message << '\n';
        }
        // errorStoreFile.txt (append)
        {
            std::ofstream fid(outPath + "\\errorStoreFile.txt", std::ios::app);
            fid << "Calibration Failure\n"
                << message << '\n';
        }
    }
    catch (...)
    {
    }
}

void Calibration::appLog(const std::string &operation, const std::string &path_d)
{
    try
    {
        std::string dir = path_d + "\\MatlabAppLog";
        std::filesystem::create_directories(dir);
        std::ofstream fid(dir + "\\appLog.txt", std::ios::app);
        auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        fid << operation << " at " << std::put_time(std::localtime(&now), "%Y-%m-%d %H:%M:%S") << '\n';
    }
    catch (...)
    {
    }
}

Eigen::MatrixXd Calibration::loadIcpPlatFid(const std::string &path_d)
{
    try
    {
        std::string file = path_d + "\\icpPlatFid.txt";
        if (!fileExists(file))
            return Eigen::MatrixXd();
        Eigen::MatrixXd M = readMatrix(file);
        // Expect columns: x y label; keep as-is
        return M;
    }
    catch (...)
    {
        return Eigen::MatrixXd();
    }
}

Eigen::Matrix<double, 3, 4> Calibration::refineProjectionMatrix(
    const Eigen::Matrix<double, 3, 4> &P_init,
    const Eigen::MatrixXd &imagePoints_homogeneous,
    const Eigen::MatrixXd &worldPoints)
{
    Eigen::Matrix<double, 3, 4> P = P_init;
    const int N = static_cast<int>(worldPoints.rows());
    if (N < 6)
        return P; // need enough points

    // Build homogeneous 3D points (4xN)
    Eigen::MatrixXd X_h(4, N);
    X_h.block(0, 0, 3, N) = worldPoints.transpose();
    X_h.row(3) = Eigen::VectorXd::Ones(N);

    const int max_iters = 100;
    const double eps = 1e-9;

    double lambda = 1e-3; // LM damping (persist across iterations)

    for (int it = 0; it < max_iters; ++it)
    {
        // Residuals (2N) and Jacobian (2N x 12)
        Eigen::VectorXd r(2 * N);
        Eigen::MatrixXd J(2 * N, 12);
        J.setZero();

        std::vector<double> res_norms;
        res_norms.reserve(N);

        for (int i = 0; i < N; ++i)
        {
            Eigen::Vector4d X = X_h.col(i);
            Eigen::Vector3d proj = P * X;
            double w = proj(2);
            if (std::abs(w) < 1e-12)
            {
                r(2 * i) = r(2 * i + 1) = 0;
                continue;
            }
            double u = proj(0) / w;
            double v = proj(1) / w;
            double u_obs = imagePoints_homogeneous(0, i);
            double v_obs = imagePoints_homogeneous(1, i);
            double du = u - u_obs;
            double dv = v - v_obs;
            r(2 * i) = du;
            r(2 * i + 1) = dv;
            res_norms.push_back(std::sqrt(du * du + dv * dv));

            // Derivatives wrt P entries: p00..p02,p03, p10..p12,p13, p20..p22,p23
            Eigen::RowVector4d Xt = X.transpose();
            double invw = 1.0 / w;
            for (int k = 0; k < 4; ++k)
            {
                double Xk = X(k);
                // row for u
                J(2 * i, 0 + k) = Xk * invw;      // p0k
                J(2 * i, 8 + k) = -u * invw * Xk; // p2k
                // row for v
                J(2 * i + 1, 4 + k) = Xk * invw;      // p1k
                J(2 * i + 1, 8 + k) = -v * invw * Xk; // p2k
            }
        }

        // Robust IRLS weights (Huber)
        auto median = [](const std::vector<double> &input)
        {
            if (input.empty())
                return 0.0;
            std::vector<double> v = input; // explicit copy
            size_t n = v.size();
            std::nth_element(v.begin(), v.begin() + n / 2, v.end());
            double m = v[n / 2];
            if (n % 2 == 0)
            {
                auto mx = *std::max_element(v.begin(), v.begin() + n / 2);
                m = (m + mx) / 2.0;
            }
            return m;
        };
        double med = median(res_norms);
        for (double &x : res_norms)
            x = std::abs(x - med);
        double mad = median(res_norms);
        double s = std::max(1e-6, 1.4826 * mad); // robust scale
        const double delta = 1.2;                // Huber threshold in pixels (stricter)

        Eigen::VectorXd wts(2 * N);
        wts.setOnes();
        for (int i = 0; i < N; ++i)
        {
            double ri = std::sqrt(r(2 * i) * r(2 * i) + r(2 * i + 1) * r(2 * i + 1));
            double t = ri / s;
            double wi = (t <= delta) ? 1.0 : (delta / t);
            wts(2 * i) = wi;
            wts(2 * i + 1) = wi;
        }

        // Form weighted normal equations (LM): (J^T W J + lambda I) dp = - J^T W r
        Eigen::MatrixXd W = wts.asDiagonal();
        Eigen::MatrixXd H = J.transpose() * W * J + lambda * Eigen::MatrixXd::Identity(12, 12);
        Eigen::VectorXd g = -J.transpose() * W * r;
        Eigen::VectorXd dp = H.ldlt().solve(g);
        if (!dp.allFinite())
            break;
        if (dp.norm() < eps)
            break;

        // Update P
        Eigen::Matrix<double, 3, 4> dP;
        dP << dp(0), dp(1), dp(2), dp(3),
            dp(4), dp(5), dp(6), dp(7),
            dp(8), dp(9), dp(10), dp(11);

        Eigen::Matrix<double, 3, 4> P_new = P + dP;

        // Evaluate new weighted cost; accept if better
        double old_cost = (W * r).squaredNorm();
        Eigen::VectorXd r_new(2 * N);
        for (int i = 0; i < N; ++i)
        {
            Eigen::Vector3d proj = P_new * X_h.col(i);
            double w = proj(2);
            if (std::abs(w) < 1e-12)
            {
                r_new(2 * i) = r_new(2 * i + 1) = 0;
                continue;
            }
            double u = proj(0) / w;
            double v = proj(1) / w;
            double u_obs = imagePoints_homogeneous(0, i);
            double v_obs = imagePoints_homogeneous(1, i);
            r_new(2 * i) = u - u_obs;
            r_new(2 * i + 1) = v - v_obs;
        }
        double new_cost = (W * r_new).squaredNorm();
        if (new_cost < old_cost)
        {
            P = P_new;
            lambda = std::max(lambda * 0.7, 1e-6);
        }
        else
        {
            lambda = std::min(lambda * 2.0, 1e3);
        }
    }

    return P;
}