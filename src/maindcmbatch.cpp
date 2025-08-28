
#include <gdcmImageReader.h>
#include <gdcmImage.h>
#include <gdcmPixelFormat.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <string>
#include <json.hpp>

#include "blob_detection.h"
#include "Center_detection.h"
#include "icp_angle.h"
#include "plate12icp.h"
#include "crop_image.h"
#include "plate12indexing.h"
#include <filesystem>
namespace fs = std::filesystem;

using namespace std;
using namespace cv;
using json = nlohmann::json;

cv::Mat readDICOM_GDCM(const std::string& filepath) {
    gdcm::ImageReader reader;
    reader.SetFileName(filepath.c_str());

    if (!reader.Read()) {
        throw std::runtime_error("Failed to read DICOM file: " + filepath);
    }

    const gdcm::Image& image = reader.GetImage();
    const unsigned int* dims = image.GetDimensions();
    int width = dims[0], height = dims[1];

    gdcm::PixelFormat pf = image.GetPixelFormat();
    int numChannels = pf.GetSamplesPerPixel();

    std::vector<char> buffer(image.GetBufferLength());
    if (!image.GetBuffer(buffer.data())) {
        throw std::runtime_error("Failed to get pixel buffer");
    }

    int cvDepth;
    switch (pf.GetScalarType()) {
        case gdcm::PixelFormat::UINT8:   cvDepth = CV_8U; break;
        case gdcm::PixelFormat::INT8:    cvDepth = CV_8S; break;
        case gdcm::PixelFormat::UINT16:  cvDepth = CV_16U; break;
        case gdcm::PixelFormat::INT16:   cvDepth = CV_16S; break;
        case gdcm::PixelFormat::UINT32:  cvDepth = CV_32S; break;
        case gdcm::PixelFormat::FLOAT32: cvDepth = CV_32F; break;
        case gdcm::PixelFormat::FLOAT64: cvDepth = CV_64F; break;
        default:
            throw std::runtime_error("Unsupported GDCM pixel format");
    }

    cv::Mat img(height, width, CV_MAKETYPE(cvDepth, numChannels), buffer.data());
    cv::Mat imgCopy = img.clone();

    double minVal, maxVal;
    cv::minMaxLoc(imgCopy, &minVal, &maxVal);
    cv::Mat img8U;
    imgCopy.convertTo(img8U, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

    return img8U;
}

vector<Point2f> extractCentroids(const Mat& img) {
    vector<Point2f> centroids;
    Mat labels, stats, centroidsMat;
    int n_labels = connectedComponentsWithStats(img, labels, stats, centroidsMat, 8, CV_32S);
    for (int i = 1; i < n_labels; ++i) {
        centroids.push_back(Point2f(
            static_cast<float>(centroidsMat.at<double>(i, 0)),
            static_cast<float>(centroidsMat.at<double>(i, 1))
        ));
    }
    return centroids;
}

// --- Draw a set of points ---
void drawPoints(Mat& img, const vector<Point2f>& points, Scalar color, int radius = 5) {
    for (const auto& pt : points) {
        if (pt.x == 0 && pt.y == 0) continue;
        circle(img, pt, radius, color, FILLED);
    }
}

int main() {
    std::string input_folder = "C:/Users/Sandeep/OneDrive/Desktop/Test/";
    std::string output_dir = "./output/";

    fs::create_directory(output_dir);

    for (const auto& entry : fs::directory_iterator(input_folder)) {
        if (entry.path().extension() != ".dcm") continue;

        std::string dicomPath = entry.path().string();
        std::string baseName = entry.path().stem().string();  // filename without extension

        std::cout << "Processing: " << dicomPath << std::endl;

        try {
            cv::Mat img8u = readDICOM_GDCM(dicomPath);

            // Image quality check with blob detection
            SimpleBlobDetector::Params params;
            params.filterByArea = true;
            params.minArea = 15;
            params.maxArea = 900;
            Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
            vector<KeyPoint> keypoints;
            detector->detect(img8u, keypoints);

            if (keypoints.size() < 5) {  // Reasonable threshold for batch processing
                std::cerr << "[WARNING] Image quality check failed for " << baseName << ": Not enough blobs detected." << std::endl;
                continue;  // Skip this file and continue with next
            }

            int xmin, ymin, xmax, ymax;

            // Crop directly in memory
            Mat cropped_input = CropImage::run(img8u, xmin, ymin, xmax, ymax);
            //cv::imwrite(output_dir + baseName + "_cropped.png", cropped_input);

            // Run blob detection directly on Mat
            BlobDetectionResult blobRes = run_blob_detection(cropped_input, output_dir);
            if (!output_dir.empty()) {
                std::string fiducialPath = output_dir + "/" + baseName + "_fiducials.png";
                cv::imwrite(fiducialPath, blobRes.finalImage);
            }
            // --- Extract Fiducials ---
            std::vector<Point2f> centers;
            std::vector<float> radii;
            for (const auto& kp : blobRes.verifiedKeypoints) {
                centers.push_back(kp.pt);
                radii.push_back(kp.size / 2.0f);
            }

            std::vector<Point2f> centroids = extractCentroids(blobRes.binaryBlobImage);
            auto [C, first_ring_balls] = Center_detection(centroids, blobRes.binaryBlobImage.rows, blobRes.binaryBlobImage.cols);

            std::vector<std::vector<float>> Z1;
            for (size_t i = 0; i < centers.size(); ++i) {
                float r = radii[i];
                if (r >= 8.0f && r <= 30.0f) {
                    float dist = norm(centers[i] - C);
                    Z1.push_back({dist, centers[i].x, centers[i].y, r});
                }
            }

            std::vector<std::vector<float>> Z2 = Z1;
            std::sort(Z2.begin(), Z2.end(), [](const auto& a, const auto& b) { return a[0] < b[0]; });

            std::vector<std::vector<float>> Z2filt;
            for (const auto& row : Z2) {
                if (row[0] > 230 && row[0] < 450) Z2filt.push_back(row);
            }
            Z2 = Z2filt;

            std::vector<float> dists;
            for (const auto& row : Z2) dists.push_back(row[0]);
            float minDist = dists.empty() ? 0 : *min_element(dists.begin(), dists.end());
            float maxDist = dists.empty() ? 1 : *max_element(dists.begin(), dists.end());

            std::vector<float> Z2norm;
            for (const auto& d : dists) Z2norm.push_back((d - minDist) / (maxDist - minDist + 1e-6f));

            std::vector<size_t> Z2idx;
            for (size_t i = 0; i < Z2norm.size(); ++i)
                if (Z2norm[i] < 0.6) Z2idx.push_back(i);
            if (Z2idx.size() <= 1) for (size_t i = 0; i < Z2norm.size(); ++i)
                if (Z2norm[i] < 0.7) Z2idx.push_back(i);
            if (Z2idx.size() <= 1) for (size_t i = 0; i < Z2norm.size(); ++i)
                if (Z2norm[i] < 0.8) Z2idx.push_back(i);

            std::vector<std::vector<float>> Z3;
            for (auto idx : Z2idx) Z3.push_back(Z2[idx]);

            std::cout << "DEBUG: Z2 size: " << Z2.size() << std::endl;
            std::cout << "DEBUG: Z2idx size: " << Z2idx.size() << std::endl;
            std::cout << "DEBUG: Z3 size: " << Z3.size() << std::endl;

            // Save ICP 2D data
            std::ofstream icpFile(output_dir + baseName + "_icp_2D.txt");
            for (const auto& row : Z3) {
                for (size_t i = 0; i < row.size(); ++i)
                    icpFile << std::fixed << std::setprecision(6) << row[i] << (i + 1 < row.size() ? " " : "\n");
            }
            icpFile.close();

            std::vector<Point2f> icpFidVec;
            for (const auto& row : Z3)
                if (row.size() >= 3) icpFidVec.emplace_back(row[1], row[2]);

            PlateFiducials plateResult = plate12icp(blobRes.binaryBlobImage, C, icpFidVec);
            plateResult = plate12withICP_post(plateResult, Z3, C);

            // Save results with labels
            std::ofstream resultFile(output_dir + baseName + "_final_plate_points.txt");
            resultFile << "# Plate 1 Points\n";
            for (const auto& pt : plateResult.final_plate1)
                resultFile << pt.x << " " << pt.y << " " << pt.label << "\n";

            resultFile << "\n# Plate 2 Points\n";
            for (const auto& pt : plateResult.final_plate2)
                resultFile << pt.x << " " << pt.y << " " << pt.label << "\n";

            resultFile << "\n# ICP Points\n";
            for (const auto& pt : plateResult.icpPlatFid)
                resultFile << pt.x << " " << pt.y << " " << pt.label << "\n";

            resultFile.close();

          // Create enhanced visualization
cv::Mat img_color;
if (cropped_input.channels() == 1) {
    cv::cvtColor(cropped_input, img_color, cv::COLOR_GRAY2BGR);
} else {
    img_color = cropped_input.clone();
}

// Draw circles from Z3 (detected blobs)
for (const auto& row : Z3) {
    cv::Point center(cvRound(row[1]), cvRound(row[2]));
    int radius = cvRound(row[3]);
    cv::circle(img_color, center, radius, cv::Scalar(128, 128, 128), 1); // Gray circles
}

// Draw center point
cv::circle(img_color, C, 4, cv::Scalar(0, 0, 255), 3);
cv::putText(img_color, "CENTER", C + cv::Point2f(10, -10),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);

// ======== MATLAB-STYLE UNIFIED NUMBERING ========

// Combine all points into one vector
std::vector<cv::Point3f> allPoints; // x, y, label
for (const auto& pt : plateResult.final_plate1)
    allPoints.emplace_back(pt.x, pt.y, pt.label);
for (const auto& pt : plateResult.final_plate2)
    allPoints.emplace_back(pt.x, pt.y, pt.label);
for (const auto& pt : plateResult.icpPlatFid)
    allPoints.emplace_back(pt.x, pt.y, pt.label);

// Draw all points with green numbering
for (const auto& pt : allPoints) {
    cv::Point2f center(pt.x, pt.y);
    std::string label = std::to_string(static_cast<int>(pt.z));

    // Calculate text size for centering
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseline);
    cv::Point2f textPos = center - cv::Point2f(textSize.width / 2.0f, -textSize.height / 2.0f);

    // Optional: white background box like MATLAB
    // cv::Rect textBox(textPos.x - 3, textPos.y - textSize.height - 3,
    //                  textSize.width + 6, textSize.height + 6);
    // cv::rectangle(img_color, textBox, cv::Scalar(255, 255, 255), -1);
    // cv::rectangle(img_color, textBox, cv::Scalar(0, 0, 0), 1);

    // Green text
    cv::putText(img_color, label, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 0), 1.8);
}

    // // ======== LEGEND ========
    // int legendY = 30;
    // cv::putText(img_color, "LEGEND:", cv::Point(10, legendY),
    //             cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    // legendY += 25;
    // cv::putText(img_color, "Green Numbers = Fiducial IDs", cv::Point(40, legendY),
    //             cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

    // ======== SAVE OUTPUT ========
    std::string indexed_output = output_dir + baseName + "_indexed_fiducials.png";
    cv::imwrite(indexed_output, img_color);

    std::cout << "=== OUTPUT FILES for " << baseName << " ===" << std::endl;
    std::cout << "Indexed image saved to: " << indexed_output << std::endl;
    std::cout << "Point coordinates saved to: " << output_dir + baseName + "_final_plate_points.txt" << std::endl;
    std::cout << "ICP 2D data saved to: " << output_dir + baseName + "_icp_2D.txt" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "[ERROR] " << e.what() << "\n";
        }
    }

    std::cout << "Batch processing complete.\n";
    return 0;
}
