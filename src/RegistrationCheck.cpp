#include "RegistrationCheck.h"
#include <algorithm>
#include <limits>
#include <vector>
#include <iostream>

static double computeRegistration(
    const Eigen::Matrix<double, 3, 4>& P,
    const Eigen::MatrixXd& W_dist_pts,
    const std::string& position,
    double r,
    const std::string& path_d,
    int rewarp) {
    try {
        std::string folder = (rewarp == 1) ? "Rewarp\\" : "";

        // Read blob image
        std::string blob_file = path_d + "\\Output\\" + position + "\\PD\\" + position + "bw.png";
        cv::Mat blob = cv::imread(blob_file, cv::IMREAD_GRAYSCALE);
        if (blob.empty()) {
            std::cerr << "Warning: Could not read blob image: " << blob_file << std::endl;
            return 10.0;
        }

        // Binarize
        cv::threshold(blob, blob, 0, 255, cv::THRESH_BINARY);

        // Connected components (8-connectivity)
        const int connectivity = 8;
        cv::Mat labels, stats, centroidsMat;
        int nLabels = cv::connectedComponentsWithStats(blob, labels, stats, centroidsMat, connectivity, CV_32S);

        // Collect centroids, skipping background
        std::vector<cv::Point2f> centres;
        centres.reserve(std::max(nLabels - 1, 0));
        for (int i = 1; i < nLabels; ++i) {
            double cx = centroidsMat.at<double>(i, 0);
            double cy = centroidsMat.at<double>(i, 1);
            centres.emplace_back(static_cast<float>(cx), static_cast<float>(cy));
        }
        if (centres.empty()) {
            std::cerr << "Warning: No centres detected in blob image." << std::endl;
            return 10.0;
        }

        // Project 3D points
        Eigen::MatrixXd projected_2d_pts = P * W_dist_pts; // 3xN
        for (int i = 0; i < projected_2d_pts.cols(); ++i) projected_2d_pts.col(i) /= projected_2d_pts(2, i);

        const int N = static_cast<int>(projected_2d_pts.cols());
        Eigen::MatrixXd projected_pts_2d(N, 2);
        projected_pts_2d.col(0) = projected_2d_pts.row(0);
        projected_pts_2d.col(1) = r - projected_2d_pts.row(1).array();

        // Distance matrix
        Eigen::MatrixXd distance_compute(N, centres.size());
        for (int i = 0; i < N; ++i) {
            const double px = projected_pts_2d(i, 0);
            const double py = projected_pts_2d(i, 1);
            for (size_t j = 0; j < centres.size(); ++j) {
                const double dx = px - centres[j].x;
                const double dy = py - centres[j].y;
                distance_compute(i, j) = std::sqrt(dx*dx + dy*dy);
            }
        }

        // Row-wise nearest neighbors
        std::vector<int> rowMinIdx(N, -1);
        std::vector<double> rowMinDist(N, std::numeric_limits<double>::infinity());
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < distance_compute.cols(); ++j) {
                double d = distance_compute(i, j);
                if (d < rowMinDist[i]) { rowMinDist[i] = d; rowMinIdx[i] = j; }
            }
        }

        // Column-wise nearest neighbors
        std::vector<int> colMinIdx(distance_compute.cols(), -1);
        std::vector<double> colMinDist(distance_compute.cols(), std::numeric_limits<double>::infinity());
        for (int j = 0; j < distance_compute.cols(); ++j) {
            for (int i = 0; i < N; ++i) {
                double d = distance_compute(i, j);
                if (d < colMinDist[j]) { colMinDist[j] = d; colMinIdx[j] = i; }
            }
        }

        // Threshold sweep
        const double t_min = 0.0, t_max = 1.5, t_step = 0.1;
        double best_err = std::numeric_limits<double>::infinity();
        double best_t = t_min;
        int k_min = static_cast<int>(std::round(t_min / t_step));
        int k_max = static_cast<int>(std::round(t_max / t_step));
        for (int k_idx = k_min; k_idx <= k_max; ++k_idx) {
            double t = std::round((static_cast<double>(k_idx) * t_step) * 10.0) / 10.0;
            std::vector<double> matchedDistances;
            matchedDistances.reserve(N);
            for (int i = 0; i < N; ++i) {
                int j = rowMinIdx[i];
                if (j >= 0 && colMinIdx[j] == i && rowMinDist[i] <= t) matchedDistances.push_back(rowMinDist[i]);
            }
            if (!matchedDistances.empty()) {
                std::sort(matchedDistances.begin(), matchedDistances.end());
                size_t ksz = matchedDistances.size();
                size_t drop = (ksz >= 10) ? static_cast<size_t>(std::floor(0.1 * ksz)) : 0;
                double sum = 0.0; size_t cnt = 0;
                for (size_t idx = 0; idx < ksz - drop; ++idx) { sum += matchedDistances[idx]; ++cnt; }
                double err = (cnt > 0) ? (sum / static_cast<double>(cnt)) : matchedDistances.back();
                if (err < best_err) { best_err = err; best_t = t; }
            }
        }

        // Hungarian
        auto hungarianAssign = [&](const Eigen::MatrixXd& cost) -> std::vector<int> {
            const int n = static_cast<int>(cost.rows());
            std::vector<double> u(n + 1, 0.0), v(n + 1, 0.0);
            std::vector<int> p(n + 1, 0), way(n + 1, 0);
            for (int i = 1; i <= n; ++i) {
                p[0] = i;
                int j0 = 0; 
                std::vector<double> minv(n + 1, std::numeric_limits<double>::infinity());
                std::vector<char> used(n + 1, false);
                do {
                    used[j0] = true;
                    int i0 = p[j0], j1 = 0;
                    double delta = std::numeric_limits<double>::infinity();
                    for (int j = 1; j <= n; ++j) if (!used[j]) {
                        double cur = cost(i0 - 1, j - 1) - u[i0] - v[j];
                        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                    }
                    for (int j = 0; j <= n; ++j) {
                        if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                        else { minv[j] -= delta; }
                    }
                    j0 = j1;
                } while (p[j0] != 0);
                do { int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
            }
            std::vector<int> row_to_col(n, -1);
            for (int j = 1; j <= n; ++j) if (p[j] > 0) row_to_col[p[j] - 1] = j - 1;
            return row_to_col;
        };

        const int Rn = N;
        const int Cn = static_cast<int>(centres.size());
        const double t = best_t;
        const double dummyPenalty = std::max(0.75 * t, 0.5);
        const int sz = Rn + Cn;
        Eigen::MatrixXd AC(sz, sz);
        AC.setZero();
        for (int i = 0; i < Rn; ++i) for (int j = 0; j < Cn; ++j) {
            double c = distance_compute(i, j);
            if (!(c <= t)) c = 10.0 * t;
            AC(i, j) = c;
        }
        for (int i = 0; i < Rn; ++i) for (int j = 0; j < Rn; ++j) AC(i, Cn + j) = dummyPenalty;
        for (int i = 0; i < Cn; ++i) for (int j = 0; j < Cn; ++j) AC(Rn + i, j) = dummyPenalty;

        std::vector<int> row_to_col = hungarianAssign(AC);

        std::vector<double> matchedDistances;
        matchedDistances.reserve(std::min(Rn, Cn));
        for (int i = 0; i < Rn; ++i) {
            int j = row_to_col[i];
            if (j >= 0 && j < Cn) {
                double d = distance_compute(i, j);
                if (d <= t) matchedDistances.push_back(d);
            }
        }

        double hung_err = std::numeric_limits<double>::infinity();
        if (!matchedDistances.empty()) {
            std::sort(matchedDistances.begin(), matchedDistances.end());
            size_t k = matchedDistances.size();
            size_t drop = (k >= 10) ? static_cast<size_t>(std::floor(0.1 * k)) : 0;
            double sum = 0.0; size_t cnt = 0;
            for (size_t idx = 0; idx < k - drop; ++idx) { sum += matchedDistances[idx]; ++cnt; }
            if (cnt > 0) hung_err = sum / static_cast<double>(cnt);
        }

        double chosen_err;
        std::string chosen_method;
        if (std::isfinite(hung_err)) { chosen_err = hung_err; chosen_method = "hungarian"; }
        else if (std::isfinite(best_err)) { chosen_err = best_err; chosen_method = "mutualNN"; }
        else { chosen_err = 10.0; chosen_method = "none"; }

        std::cout << "registrationCheck errors (px) -> "
                  << "mutualNN: " << (std::isfinite(best_err) ? best_err : -1.0)
                  << " (t=" << best_t << ")"
                  << ", hungarian: " << (std::isfinite(hung_err) ? hung_err : -1.0)
                  << ", chosen: " << chosen_err << " [" << chosen_method << "]"
                  << std::endl;

        return chosen_err;
    } catch (const std::exception& e) {
        std::cerr << "Error in registration check: " << e.what() << std::endl;
        return 10.0;
    }
}

double registrationCheck(
    const Eigen::Matrix<double, 3, 4>& P,
    const Eigen::MatrixXd& W_dist_pts,
    const std::string& position,
    double r,
    const std::string& path_d,
    int rewarp) {
    return computeRegistration(P, W_dist_pts, position, r, path_d, rewarp);
}