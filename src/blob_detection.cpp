#include "blob_detection.h"
#include <fstream>
#include <filesystem>
#include "LC.hpp"

namespace fs = std::filesystem;

namespace {

// === Apply CLAHE + Bilateral Denoising ===
cv::Mat applyCLAHEandDenoise(const cv::Mat& inputGray) {
    cv::Mat claheResult, denoised;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.1, cv::Size(3, 3));
    clahe->apply(inputGray, claheResult);
    cv::bilateralFilter(claheResult, denoised, 6, 65, 65);
    return denoised;
}

// === Blob Detection with SimpleBlobDetector ===
std::vector<cv::KeyPoint> detectBlobs(const cv::Mat& input, float minInertiaRatio) {
    cv::SimpleBlobDetector::Params params;
    params.minThreshold = 1;
    params.maxThreshold = 255;
    params.thresholdStep = 4;
    params.filterByArea = true;
    params.minArea = 20;   
    params.maxArea = 800;  
    params.filterByCircularity = true;
    params.minCircularity = 0.8;
    params.filterByInertia = true;
    params.minInertiaRatio = minInertiaRatio;
    params.filterByConvexity = true;
    params.minConvexity = 0.9;

    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(input, keypoints);
    return keypoints;
}

// === Verify Blob Shape by Contour Circularity ===
bool verifyByContour(const cv::Mat& image, const cv::KeyPoint& kp) {
    int radius = static_cast<int>(kp.size / 2);
    cv::Rect roi(cv::Point(kp.pt.x - radius, kp.pt.y - radius), cv::Size(radius * 2, radius * 2));
    roi = roi & cv::Rect(0, 0, image.cols, image.rows);

    cv::Mat patch = image(roi).clone();
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(patch, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        double perimeter = cv::arcLength(contour, true);
        if (perimeter == 0) continue;
        double circularity = 4 * CV_PI * area / (perimeter * perimeter);
        if (circularity > 0.7 && area > 5) return true;
    }
    return false;
}

// === Measure Accurate Blob Radius via minEnclosingCircle ===
bool measureBlobRadius(const cv::Mat& binaryImg, const cv::KeyPoint& kp, float& accurateRadius, float& circularity) {
    cv::Rect roi(cv::Point(kp.pt.x - kp.size / 2, kp.pt.y - kp.size / 2),
                 cv::Size(static_cast<int>(kp.size), static_cast<int>(kp.size)));
    roi = roi & cv::Rect(0, 0, binaryImg.cols, binaryImg.rows);

    cv::Mat patch = binaryImg(roi).clone();
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(patch, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) return false;

    // Find largest contour in ROI
    size_t maxIdx = 0;
    double maxArea = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        double a = cv::contourArea(contours[i]);
        if (a > maxArea) {
            maxArea = a;
            maxIdx = i;
        }
    }

    // Fit circle
    cv::Point2f center;
    float radius;
    cv::minEnclosingCircle(contours[maxIdx], center, radius);

    // Compute circularity
    double perimeter = cv::arcLength(contours[maxIdx], true);
    circularity = (perimeter > 0) ? (4 * CV_PI * maxArea / (perimeter * perimeter)) : 0.0f;

    accurateRadius = radius;
    return true;
}

} // namespace


// === Core Detection Implementation (shared) ===
static BlobDetectionResult run_blob_detection_impl(const cv::Mat& inputBGRorGray,
                                                   const std::string& outputDir,
                                                   float minInertiaRatio,
                                                   bool saveOutputs) {
    BlobDetectionResult result;

    if (saveOutputs && !outputDir.empty()) {
        result.finalImagePath = outputDir + "/fiducials.png";
        result.binaryBlobPath = outputDir + "/binary_blobs.png";
        result.pixelCsvPath   = outputDir + "/blob_pixels.csv";
    }

    if (inputBGRorGray.empty())
        throw std::runtime_error("Input image is empty.");

    // --- Ensure grayscale ---
    cv::Mat gray;
    if (inputBGRorGray.channels() == 1) {
        gray = inputBGRorGray.clone();
    } else {
        cv::cvtColor(inputBGRorGray, gray, cv::COLOR_BGR2GRAY);
    }

    // Auto inertia ratio if unset
    double meanBrightness = cv::mean(gray)[0];
    if (minInertiaRatio < 0)
        minInertiaRatio = (meanBrightness < 100.0) ? 0.4f : 0.7f;

    // Enhance
    cv::Mat contrastImg = localcontrast(inputBGRorGray, 0.1);
    cv::Mat claheOrig = applyCLAHEandDenoise(gray);

    cv::Mat grayContrast;
    if (contrastImg.channels() == 1)
        grayContrast = contrastImg;
    else
        cv::cvtColor(contrastImg, grayContrast, cv::COLOR_BGR2GRAY);

    cv::Mat claheContrast = applyCLAHEandDenoise(grayContrast);

    // Detect
    auto keypointsOrig = detectBlobs(claheOrig, minInertiaRatio);
    auto keypointsContrast = detectBlobs(claheContrast, minInertiaRatio);

    std::vector<cv::KeyPoint> selectedKeypoints;
    cv::Mat selectedClahe = claheOrig;
    if (keypointsContrast.size() > keypointsOrig.size()) {
        selectedKeypoints = keypointsContrast;
        selectedClahe = claheContrast;
    } else {
        selectedKeypoints = keypointsOrig;
    }

    std::cout << "[Selected] Initial Keypoints: " << selectedKeypoints.size() << std::endl;

    // Create binary mask
    cv::Mat binaryBlobImage = cv::Mat::zeros(gray.size(), CV_8UC1);
    for (const auto& kp : selectedKeypoints) {
        cv::circle(binaryBlobImage, kp.pt, static_cast<int>(kp.size / 2), cv::Scalar(255), -1);
    }

    // Verify by contour
    std::vector<cv::KeyPoint> verifiedKeypoints;
    for (const auto& kp : selectedKeypoints) {
        if (verifyByContour(binaryBlobImage, kp)) {
            verifiedKeypoints.push_back(kp);
        }
    }

    std::cout << "[Final] Verified Keypoints: " << verifiedKeypoints.size() << std::endl;

    // Draw
    cv::drawKeypoints(inputBGRorGray, verifiedKeypoints, result.finalImage, cv::Scalar(0, 0, 255),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    for (size_t i = 0; i < verifiedKeypoints.size(); ++i) {
        cv::putText(result.finalImage, std::to_string(i + 1),
                    verifiedKeypoints[i].pt + cv::Point2f(6, -6),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 255), 1);
    }

    result.binaryBlobImage = binaryBlobImage;
    result.verifiedKeypoints = verifiedKeypoints;

    // Save only if requested
    if (saveOutputs && !outputDir.empty()) {
        cv::imwrite(result.finalImagePath, result.finalImage);
        cv::imwrite(result.binaryBlobPath, result.binaryBlobImage);

        // CSV export (unchanged)
        std::ofstream pixelFile(result.pixelCsvPath);
        pixelFile << "Index,X,Y,Area,Perimeter,CentroidX,CentroidY,Radius,Circularity\n";
        int index = 1;

        for (const auto& kp : verifiedKeypoints) {
            float accurateRadius = 0.0f, circularity = 0.0f;
            if (measureBlobRadius(binaryBlobImage, kp, accurateRadius, circularity)) {
                int radius = static_cast<int>(accurateRadius);
                cv::Rect roi(cv::Point(kp.pt.x - radius, kp.pt.y - radius), cv::Size(radius * 2, radius * 2));
                roi = roi & cv::Rect(0, 0, binaryBlobImage.cols, binaryBlobImage.rows);

                cv::Mat patch = binaryBlobImage(roi).clone();
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(patch, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                for (const auto& contour : contours) {
                    double area = cv::contourArea(contour);
                    if (area < 5) continue;
                    double perimeter = cv::arcLength(contour, true);
                    cv::Moments mu = cv::moments(contour);
                    if (mu.m00 == 0) continue;
                    cv::Point2f centroid(mu.m10 / mu.m00 + roi.x, mu.m01 / mu.m00 + roi.y);

                    pixelFile << index << "," << kp.pt.x << "," << kp.pt.y << ","
                              << area << "," << perimeter << ","
                              << centroid.x << "," << centroid.y << ","
                              << accurateRadius << "," << circularity << "\n";
                    ++index;
                    break;
                }
            }
        }

        pixelFile.close();
    }

    return result;
}

// === Public API ===
BlobDetectionResult run_blob_detection(const std::string& imgPath,
                                       const std::string& outputDir,
                                       float minInertiaRatio) {
    // Always load grayscale from disk
    cv::Mat input = cv::imread(imgPath, cv::IMREAD_COLOR);
    if (input.empty()) throw std::runtime_error("Failed to load image: " + imgPath);
    return run_blob_detection_impl(input, outputDir, minInertiaRatio, /*saveOutputs=*/true);
}

BlobDetectionResult run_blob_detection(const cv::Mat& inputImage,
                                       const std::string& outputDir,
                                       float minInertiaRatio) {
    // Already have cv::Mat (BGR or Gray)
    return run_blob_detection_impl(inputImage, outputDir, minInertiaRatio, /*saveOutputs=*/true);
}