#pragma once
#include <opencv2/opencv.hpp>

inline cv::Mat localcontrast(const cv::Mat &input, double edgeThreshold = 0.3, double amount = 0.5,
                             double alpha_scale = 1.5, double beta = 18.5) {
    cv::Mat src;
    input.convertTo(src, CV_32F, 1.0 / 255.0);

    cv::Mat gray;
    if (src.channels() == 3)
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else
        gray = src.clone();

    double alpha = (amount > 0) ? 1.0 - 0.99 * amount : 1.0 - 99.0 * amount;

    cv::Mat smooth;
    cv::bilateralFilter(gray, smooth, -1, edgeThreshold * 50.0, edgeThreshold * 50.0);
    cv::Mat detail = gray - smooth;

    cv::Mat modified = (amount >= 0) ? gray + (1.0 - alpha) * detail : smooth + alpha * detail;

    cv::Mat result;
    cv::threshold(modified, result, 1.0, 1.0, cv::THRESH_TRUNC);
    cv::threshold(result, result, 0.0, 0.0, cv::THRESH_TOZERO);

    if (src.channels() == 3) {
        cv::Mat ycrcb;
        cv::cvtColor(src, ycrcb, cv::COLOR_BGR2YCrCb);
        std::vector<cv::Mat> channels;
        cv::split(ycrcb, channels);
        result.convertTo(channels[0], CV_32F);
        cv::merge(channels, ycrcb);
        cv::cvtColor(ycrcb, result, cv::COLOR_YCrCb2BGR);
    }

    result.convertTo(result, CV_8U, 255.0);
    cv::Mat final;
    result.convertTo(final, -1, alpha_scale, beta);
    return final;
}
