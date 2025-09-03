#include "crop_image.h"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

cv::Mat CropImage::run(const cv::Mat &img_in, int &xmin, int &ymin, int &xmax, int &ymax)
{
    cv::Mat img = img_in.clone();
    int col = img.cols;
    int row = img.rows;
    int ch = img.channels();

    // === Normalize to [0,1] like MATLAB rescale ===
    cv::Mat temp;
    img.convertTo(temp, CV_32F); // keep all channels as float
    double minVal, maxVal;
    cv::minMaxLoc(temp.reshape(1), &minVal, &maxVal);
    temp = (temp - minVal) / (maxVal - minVal);

    // === Diagonal white-strip check ===
    int i = 1;
    if (temp.channels() == 1)
    {
        if (temp.at<float>(0, 0) >= 1.0f)
        {
            for (int j = 0; j < std::min(row, col); ++j)
            {
                if (temp.at<float>(j, j) == 0.0f)
                {
                    i = j; // MATLAB sets i=j
                    break;
                }
            }
        }
        else
        {
            i = 5;
        }
    }
    else
    {
        if (temp.at<cv::Vec3f>(0, 0)[0] >= 1.0f)
        {
            for (int j = 0; j < std::min(row, col); ++j)
            {
                if (temp.at<cv::Vec3f>(j, j)[0] == 0.0f)
                {
                    i = j; // MATLAB sets i=j
                    break;
                }
            }
        }
        else
        {
            i = 5;
        }
    }

    // === Remove white strip (all channels) ===
    for (int y = 0; y < row; ++y)
    {
        for (int x = 0; x < col; ++x)
        {
            if (x < i || y < i || x > (col - i) || y > (row - i))
            {
                if (ch == 1)
                {
                    img.at<uchar>(y, x) = 0;
                }
                else
                {
                    cv::Vec3b &pix = img.at<cv::Vec3b>(y, x);
                    pix[0] = pix[1] = pix[2] = 0;
                }
            }
        }
    }

    // === "imadjust" equivalent: min–max stretch to [0,255] ===
    cv::Mat changed;
    img.convertTo(changed, CV_32F);
    double minA, maxA;
    cv::minMaxLoc(changed.reshape(1), &minA, &maxA);
    changed = (changed - minA) / (maxA - minA);
    changed.convertTo(changed, CV_8U, 255);

    // === Ensure single-channel for thresholding ===
    cv::Mat gray;
    if (changed.channels() > 1)
    {
        cv::cvtColor(changed, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray = changed;
    }

    // === Binary mask: > 0.1 (≈ 25 in 8-bit) ===
    cv::Mat mask_u8;
    cv::threshold(gray, mask_u8, 25, 255, cv::THRESH_BINARY);

    // === Find largest contour ===
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask_u8, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty())
    {
        std::cerr << "[CropImage] No regions found!" << std::endl;
        xmin = ymin = xmax = ymax = 0;
        return img_in.clone();
    }

    double maxArea = 0;
    int index = 0;
    for (size_t k = 0; k < contours.size(); ++k)
    {
        double area = cv::contourArea(contours[k]);
        if (area > maxArea)
        {
            maxArea = area;
            index = static_cast<int>(k);
        }
    }

    cv::Rect base = cv::boundingRect(contours[index]);

    // === Expand bounding box by z ===
    int z = 10;
    int x1 = std::max(0, base.x - z);
    int y1 = std::max(0, base.y - z);
    int x2 = std::min(col, base.x + base.width + z);
    int y2 = std::min(row, base.y + base.height + z);

    xmin = x1;
    ymin = y1;
    xmax = x2;
    ymax = y2;

    cv::Rect roi(x1, y1, x2 - x1, y2 - y1);

    // === Crop original input (preserves channels) ===
    return img_in(roi).clone();
}
