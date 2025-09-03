#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct BlobDetectionResult
{
    std::vector<cv::KeyPoint> verifiedKeypoints; // Final verified blobs
    cv::Mat binaryBlobImage;                     // Binary mask of blobs
    cv::Mat finalImage;                          // Image annotated with blobs
};

/**
 * @brief Run full fiducial/blob detection pipeline from image path.
 *
 * @param imgPath           Path to input image file (e.g., JPEG).
 * @param outputDir         Folder path to save outputs (image, mask, CSV).
 * @param minInertiaRatio   Optional inertia ratio (set -1 for auto-adjust).
 * @return BlobDetectionResult struct with keypoints, images, paths.
 */
BlobDetectionResult run_blob_detection(
    const std::string &imgPath,
    const std::string &outputDir,
    float minInertiaRatio = -1.0f);

/**
 * @brief Run full fiducial/blob detection pipeline from cv::Mat input image.
 *
 * @param inputImage        Input image matrix (grayscale or BGR).
 * @param outputDir         Folder path to save outputs (image, mask, CSV).
 * @param minInertiaRatio   Optional inertia ratio (set -1 for auto-adjust).
 * @return BlobDetectionResult struct with keypoints, images, paths.
 */
BlobDetectionResult run_blob_detection(
    const cv::Mat &inputImage,
    const std::string &outputDir,
    float minInertiaRatio = -1.0f);