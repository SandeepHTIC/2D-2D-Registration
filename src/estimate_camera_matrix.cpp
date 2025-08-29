#include "estimate_camera_matrix.h"
#include <opencv2/opencv.hpp>
#include <cmath>

std::pair<Eigen::Matrix<double, 3, 4>, std::vector<double>> estimate_camera_matrix(
    const Eigen::MatrixXd& imagePoints,
    const Eigen::MatrixXd& worldPoints)
{
    // Validate sizes
    CV_Assert(imagePoints.cols() == 2 && worldPoints.cols() == 3);
    CV_Assert(imagePoints.rows() == worldPoints.rows());
    int M = static_cast<int>(imagePoints.rows());

    // Convert Eigen -> OpenCV
    cv::Mat imgPts(M, 2, CV_64F);
    cv::Mat wPts(M, 3, CV_64F);
    for (int i = 0; i < M; ++i) {
        imgPts.at<double>(i, 0) = imagePoints(i, 0);
        imgPts.at<double>(i, 1) = imagePoints(i, 1);
        wPts.at<double>(i, 0) = worldPoints(i, 0);
        wPts.at<double>(i, 1) = worldPoints(i, 1);
        wPts.at<double>(i, 2) = worldPoints(i, 2);
    }

    // Build A matrix (2M x 12) for DLT
    cv::Mat A(2 * M, 12, CV_64F, cv::Scalar(0));
    for (int i = 0; i < M; ++i) {
        double X = wPts.at<double>(i, 0);
        double Y = wPts.at<double>(i, 1);
        double Z = wPts.at<double>(i, 2);
        double u = imgPts.at<double>(i, 0);
        double v = imgPts.at<double>(i, 1);

        // Row for u
        A.at<double>(2 * i, 0) = X;
        A.at<double>(2 * i, 1) = Y;
        A.at<double>(2 * i, 2) = Z;
        A.at<double>(2 * i, 3) = 1.0;
        A.at<double>(2 * i, 8) = -u * X;
        A.at<double>(2 * i, 9) = -u * Y;
        A.at<double>(2 * i, 10) = -u * Z;
        A.at<double>(2 * i, 11) = -u;

        // Row for v
        A.at<double>(2 * i + 1, 4) = X;
        A.at<double>(2 * i + 1, 5) = Y;
        A.at<double>(2 * i + 1, 6) = Z;
        A.at<double>(2 * i + 1, 7) = 1.0;
        A.at<double>(2 * i + 1, 8) = -v * X;
        A.at<double>(2 * i + 1, 9) = -v * Y;
        A.at<double>(2 * i + 1, 10) = -v * Z;
        A.at<double>(2 * i + 1, 11) = -v;
    }

    // Solve using SVD
    cv::SVD svd(A, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    cv::Mat P_vec = svd.vt.row(11).t();

    // Reshape to 3x4 camera matrix
    cv::Mat P_cv = P_vec.reshape(1, 3);
    if (P_cv.at<double>(2, 3) < 0) {
        P_cv = -P_cv;
    }

    // Compute reprojection errors
    std::vector<double> reprojectionErrors;
    reprojectionErrors.reserve(M);
    for (int i = 0; i < M; ++i) {
        cv::Mat Xw = (cv::Mat_<double>(4, 1) <<
            wPts.at<double>(i, 0),
            wPts.at<double>(i, 1),
            wPts.at<double>(i, 2),
            1.0);
        cv::Mat x = P_cv * Xw; // 3x1
        double u_proj = x.at<double>(0, 0) / x.at<double>(2, 0);
        double v_proj = x.at<double>(1, 0) / x.at<double>(2, 0);
        double u_obs = imgPts.at<double>(i, 0);
        double v_obs = imgPts.at<double>(i, 1);
        double err = std::sqrt((u_proj - u_obs) * (u_proj - u_obs) + (v_proj - v_obs) * (v_proj - v_obs));
        reprojectionErrors.push_back(err);
    }

    // Convert back to Eigen 3x4
    Eigen::Matrix<double, 3, 4> P;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            P(r, c) = P_cv.at<double>(r, c);
        }
    }

    return {P, reprojectionErrors};
}