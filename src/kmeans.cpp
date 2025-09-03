#include "kmeans.h"
#include <opencv2/core.hpp>

std::vector<int> kmeans(const std::vector<float> &data, int k)
{
    cv::Mat samples(data.size(), 1, CV_32F);
    for (size_t i = 0; i < data.size(); ++i)
    {
        samples.at<float>(i, 0) = data[i];
    }
    cv::Mat labels;
    cv::kmeans(samples, k, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
               3, cv::KMEANS_PP_CENTERS);
    std::vector<int> result;
    for (int i = 0; i < labels.rows; ++i)
    {
        result.push_back(labels.at<int>(i, 0));
    }
    return result;
}
