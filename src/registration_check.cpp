#include "registration_check.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <limits>
#include <cmath>
#include <fstream>
#include <iostream>

static bool file_exists_local(const std::string& filename) {
    std::ifstream f(filename);
    return f.good();
}

double registration_check(
    const Eigen::Matrix<double, 3, 4>& P,
    const Eigen::MatrixXd& W_dist_pts,
    const std::string& position,
    double r,
    const std::string& path_d,
    int rewarp)
{
    try {
        std::string folder = (rewarp == 1) ? "Rewarp\\" : "";

        // Read blob image
        std::string blob_file = path_d + "\\Output\\" + position + "\\PD\\" + position + "bw.png";
        cv::Mat blob = cv::imread(blob_file, cv::IMREAD_GRAYSCALE);
        // Debug log file
        try {
            std::string dbg = path_d + "\\Output\\" + position + "\\debug_calib.txt";
            std::ofstream df(dbg, std::ios::app);
            if (df.is_open()) df << "blob_file_used=" << blob_file << "\n";
        } catch(...) {}

        if (blob.empty()) {
            std::cerr << "Warning: Could not read blob image: " << blob_file << std::endl;
            return 10.0;
        }

        // Binarize
        cv::threshold(blob, blob, 0, 255, cv::THRESH_BINARY);

        // Connected components
        const int connectivity = 4;
        cv::Mat labels, stats, centroidsMat;
        int nLabels = cv::connectedComponentsWithStats(blob, labels, stats, centroidsMat, connectivity, CV_32S);

        const int rows_img = blob.rows;
        const int cols_img = blob.cols;
        const double min_area = 25.0;

        size_t kept_before_border = 0;
        std::vector<cv::Point2f> centres;
        centres.reserve(std::max(nLabels - 1, 0));
        for (int i = 1; i < nLabels; ++i) { // skip background
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area < min_area) continue;
            ++kept_before_border;
            int left = stats.at<int>(i, cv::CC_STAT_LEFT);
            int top = stats.at<int>(i, cv::CC_STAT_TOP);
            int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
            int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
            int right = left + width;
            int bottom = top + height;
            if (left <= 0 || top <= 0 || right >= cols_img || bottom >= rows_img) {
                continue;
            }
            double cx = centroidsMat.at<double>(i, 0);
            double cy = centroidsMat.at<double>(i, 1);
            centres.emplace_back(static_cast<float>(cx), static_cast<float>(cy));
        }

        if (centres.empty()) {
            std::cerr << "Warning: No centres detected in blob image." << std::endl;
            return 10.0;
        }

        // Project 3D points to 2D (homogeneous), then normalize
        Eigen::MatrixXd projected_2d_pts = P * W_dist_pts; // 3xN
        for (int i = 0; i < projected_2d_pts.cols(); ++i) {
            projected_2d_pts.col(i) /= projected_2d_pts(2, i);
        }

        // Extract 2D coordinates and flip y-coordinate
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

        // Find pairs under threshold
        std::vector<std::pair<int, int>> actualSubscripts;
        const double thresh = 1.0;
        for (int i = 0; i < N; ++i) {
            for (size_t j = 0; j < centres.size(); ++j) {
                if (distance_compute(i, j) < thresh) {
                    actualSubscripts.push_back({i, static_cast<int>(j)});
                }
            }
        }

        if (!actualSubscripts.empty()) {
            double sum = 0.0;
            for (const auto& sub : actualSubscripts) {
                sum += distance_compute(sub.first, sub.second);
            }
            return sum / static_cast<double>(actualSubscripts.size());
        }

        // Fallback: mean row minimum
        double sum_min = 0.0;
        for (int i = 0; i < N; ++i) {
            double min_dist = std::numeric_limits<double>::max();
            for (int j = 0; j < distance_compute.cols(); ++j) {
                if (distance_compute(i, j) < min_dist) min_dist = distance_compute(i, j);
            }
            sum_min += min_dist;
        }
        return sum_min / static_cast<double>(N);

    } catch (const std::exception& e) {
        std::cerr << "Error in registration check: " << e.what() << std::endl;
        return 10.0;
    }
}