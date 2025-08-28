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
#include <filesystem>

#include "blob_detection.h"
#include "Center_detection.h"
#include "icp_angle.h"
#include "plate12icp.h"
#include "crop_image.h"
#include "plate12indexing.h"
#include "Calibration.h"

using namespace std;
using namespace cv;
namespace fs = std::filesystem;
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

// --- Save 2D points for calibration ---
void save2DPoints(const PlateFiducials& plateResult, const string& outputPath, double imageHeight) {
    ofstream pdFile(outputPath);
    if (!pdFile.is_open()) {
        cerr << "[ERROR] Cannot create 2D points file: " << outputPath << endl;
        return;
    }
    
    pdFile << fixed << setprecision(6);
    
    // Save plate1 points
    for (const auto& pt : plateResult.final_plate1) {
        // Format: x y confidence_score
        // Flip y coordinate back to image coordinate system
        double y_flipped = imageHeight - pt.y;
        pdFile << pt.x << " " << y_flipped << " " << 1.0 << "\n";
    }
    
    // Save plate2 points
    for (const auto& pt : plateResult.final_plate2) {
        double y_flipped = imageHeight - pt.y;
        pdFile << pt.x << " " << y_flipped << " " << 1.0 << "\n";
    }
    
    // Save ICP fiducial points
    for (const auto& pt : plateResult.icpPlatFid) {
        double y_flipped = imageHeight - pt.y;
        pdFile << pt.x << " " << y_flipped << " " << 1.0 << "\n";
    }
    
    pdFile.close();
    cout << "[INFO] 2D points saved to: " << outputPath << endl;
}

// --- Load calibration parameters from JSON ---
bool loadCalibrationParams(const json& inputJson, cv::Mat& C2R, cv::Mat& C2D, cv::Mat& W, cv::Mat& Dpts) {
    try {
        // Load Camera-to-Reference transformation (C2R) from Marker_Reference
        if (inputJson.contains("Marker_Reference")) {
            auto markerRef = inputJson["Marker_Reference"];
            if (markerRef.contains("tx") && markerRef.contains("ty") && markerRef.contains("tz") && 
                markerRef.contains("Rotation") && markerRef["Rotation"].is_array() && markerRef["Rotation"].size() == 4) {
                
                C2R = cv::Mat(7, 1, CV_64F);
                C2R.at<double>(0, 0) = markerRef["tx"];
                C2R.at<double>(1, 0) = markerRef["ty"];
                C2R.at<double>(2, 0) = markerRef["tz"];
                C2R.at<double>(3, 0) = markerRef["Rotation"][0];  // qx
                C2R.at<double>(4, 0) = markerRef["Rotation"][1];  // qy
                C2R.at<double>(5, 0) = markerRef["Rotation"][2];  // qz
                C2R.at<double>(6, 0) = markerRef["Rotation"][3];  // qw
            } else {
                cerr << "[ERROR] Invalid Marker_Reference format in JSON" << endl;
                return false;
            }
        } else {
            cerr << "[ERROR] Missing Marker_Reference parameter in JSON" << endl;
            return false;
        }
        
        // Load Camera-to-Detector transformation (C2D) from Marker_DD
        if (inputJson.contains("Marker_DD")) {
            auto markerDD = inputJson["Marker_DD"];
            if (markerDD.contains("tx") && markerDD.contains("ty") && markerDD.contains("tz") && 
                markerDD.contains("Rotation") && markerDD["Rotation"].is_array() && markerDD["Rotation"].size() == 4) {
                
                C2D = cv::Mat(7, 1, CV_64F);
                C2D.at<double>(0, 0) = markerDD["tx"];
                C2D.at<double>(1, 0) = markerDD["ty"];
                C2D.at<double>(2, 0) = markerDD["tz"];
                C2D.at<double>(3, 0) = markerDD["Rotation"][0];  // qx
                C2D.at<double>(4, 0) = markerDD["Rotation"][1];  // qy
                C2D.at<double>(5, 0) = markerDD["Rotation"][2];  // qz
                C2D.at<double>(6, 0) = markerDD["Rotation"][3];  // qw
            } else {
                cerr << "[ERROR] Invalid Marker_DD format in JSON" << endl;
                return false;
            }
        } else {
            cerr << "[ERROR] Missing Marker_DD parameter in JSON" << endl;
            return false;
        }
        
        // Load World coordinate points (W) from CMM_WorldPoints
        if (inputJson.contains("CMM_WorldPoints") && inputJson["CMM_WorldPoints"].is_array()) {
            auto worldPts = inputJson["CMM_WorldPoints"];
            int numPoints = worldPts.size();
            if (numPoints > 0 && worldPts[0].is_array() && worldPts[0].size() == 3) {
                W = cv::Mat(numPoints, 3, CV_64F);
                for (int i = 0; i < numPoints; i++) {
                    for (int j = 0; j < 3; j++) {
                        W.at<double>(i, j) = worldPts[i][j];
                    }
                }
            } else {
                cerr << "[ERROR] Invalid CMM_WorldPoints format in JSON" << endl;
                return false;
            }
        } else {
            cerr << "[ERROR] Missing CMM_WorldPoints parameter in JSON" << endl;
            return false;
        }
        
        // Load Detector points (Dpts) from CMM_Dist_pts
        if (inputJson.contains("CMM_Dist_pts") && inputJson["CMM_Dist_pts"].is_array()) {
            auto detPts = inputJson["CMM_Dist_pts"];
            int numDetPts = detPts.size();
            if (numDetPts > 0 && detPts[0].is_array() && detPts[0].size() == 3) {
                Dpts = cv::Mat(numDetPts, 3, CV_64F);
                for (int i = 0; i < numDetPts; i++) {
                    for (int j = 0; j < 3; j++) {
                        Dpts.at<double>(i, j) = detPts[i][j];
                    }
                }
            } else {
                // Create empty detector points matrix if format is invalid
                Dpts = cv::Mat::zeros(0, 3, CV_64F);
            }
        } else {
            // Create empty detector points matrix if missing
            Dpts = cv::Mat::zeros(0, 3, CV_64F);
        }
        
        cout << "[INFO] Calibration parameters loaded successfully:" << endl;
        
        return true;
    } catch (const exception& e) {
        cerr << "[ERROR] Exception loading calibration parameters: " << e.what() << endl;
        return false;
    }
}

int main() {
    string jsonPath = "C:\\Users\\Sandeep\\OneDrive\\Desktop\\HTIC\\SCN 7\\inputLP.json";
    ifstream jsonFile(jsonPath);
    if (!jsonFile) {
        cerr << "❌ Failed to open input JSON file.\n";
        return -1;
    }

    json inputJson;
    jsonFile >> inputJson;

    string dicomPath = inputJson["Image"];
    string position = inputJson["Type"];
    string base_path = "C:/Users/Sandeep/OneDrive/Desktop/2D2D Project";
    string output_dir = base_path + "/Output/" + position;
    fs::create_directories(output_dir);
    fs::create_directories(output_dir + "/PD");

    cv::Mat img8u;
    try {
        img8u = readDICOM_GDCM(dicomPath);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }

    SimpleBlobDetector::Params params;
    params.filterByArea = true;
    params.minArea = 15;
    params.maxArea = 900;
    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    vector<KeyPoint> keypoints;
    detector->detect(img8u, keypoints);

    if (keypoints.size() < 5) {
        inputJson["CropRoi"] = json::array();
        inputJson["Status"] = "Failure";
        inputJson["ErrorMessage"] = "IMAGE QUALITY NOT ADEQUATE";
        ofstream errFile(output_dir + "/output_" + position + ".json");
        errFile << setw(4) << inputJson << endl;
        return -1;
    }

    int xmin, ymin, xmax, ymax;
    Mat cropped_input = CropImage::run(img8u, xmin, ymin, xmax, ymax);
    double aspectRatio = static_cast<double>(cropped_input.rows) / cropped_input.cols;
    if (aspectRatio < 0.79 || aspectRatio > 1.1) {
        inputJson["CropRoi"] = json::array();
        inputJson["Status"] = "Failure";
        inputJson["ErrorMessage"] = "CROPPED IMAGE ASPECT RATIO OUT OF RANGE";
        ofstream errFile(output_dir + "/output_" + position + ".json");
        errFile << setw(4) << inputJson << endl;
        return -1;
    }

    cv::imwrite(output_dir + "/cropped_output.png", cropped_input);
    BlobDetectionResult blobRes = run_blob_detection(cropped_input, output_dir);

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
    
    std::ofstream icpFile(output_dir + "/icp_2D.txt");
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

    // Save results
    std::ofstream resultFile(output_dir + "/final_plate_points.txt");
    for (const auto& pt : plateResult.final_plate1) {
        resultFile << std::fixed << std::setprecision(6)
                   << pt.x << " " << pt.y << " " << pt.label << "\n";
    }
    for (const auto& pt : plateResult.final_plate2) {
        resultFile << std::fixed << std::setprecision(6)
                   << pt.x << " " << pt.y << " " << pt.label << "\n";
    }
    for (const auto& pt : plateResult.icpPlatFid) {
        resultFile << std::fixed << std::setprecision(6)
                   << pt.x << " " << pt.y << " " << pt.label << "\n";
    }
    resultFile.close();

    // Save 2D points for calibration
    string pd2DFile = output_dir + "/PD/" + position + "_2D.txt";
    save2DPoints(plateResult, pd2DFile, cropped_input.rows);

    // ===== CALIBRATION SECTION =====
    cv::Mat C2R, C2D, W, Dpts;
    if (!loadCalibrationParams(inputJson, C2R, C2D, W, Dpts)) {
        inputJson["Status"] = "Failure";
        inputJson["ErrorMessage"] = "MISSING CALIBRATION PARAMETERS";
        ofstream errFile(output_dir + "/output_" + position + ".json");
        errFile << setw(4) << inputJson << endl;
        return -1;
    }
    
    cv::Mat js = cv::Mat::zeros(1, 1, CV_64F);
    double r = static_cast<double>(cropped_input.rows);
    int rewarp = 0;
    
    try {
        CalibrationResult calibResult = Calibration(
            position, C2R, C2D, W, js, Dpts, r, base_path, rewarp
        );
        
        // ESSENTIAL RESULTS ONLY
        cout << "Center: (" << C.x << ", " << C.y << ")" << endl;
        cout << "Final Plate1 size: " << plateResult.final_plate1.size() << endl;
        cout << "Final Plate2 size: " << plateResult.final_plate2.size() << endl;
        cout << "Mean reprojection error: " << cv::mean(cv::Mat())[0] << " pixels" << endl; // Will be calculated in calibration
        cout << "Registration error (RPE): " << calibResult.RPE << endl;
        
        inputJson["Status"] = "Success";
        inputJson["CalibrationRPE"] = calibResult.RPE;
        
        json calibMatrix = json::array();
        for (int i = 0; i < 4; i++) {
            json row = json::array();
            for (int j = 0; j < 4; j++) {
                row.push_back(calibResult.resultMatrix.at<double>(i, j));
            }
            calibMatrix.push_back(row);
        }
        inputJson["CalibrationMatrix"] = calibMatrix;
        
    } catch (const exception& e) {
        inputJson["Status"] = "Failure";
        inputJson["ErrorMessage"] = "CALIBRATION_FAILED: " + string(e.what());
        inputJson["CalibrationRPE"] = 10.0;
    }

    // Create visualization
    cv::Mat img_color;
    if (cropped_input.channels() == 1) {
        cv::cvtColor(cropped_input, img_color, cv::COLOR_GRAY2BGR);
    } else {
        img_color = cropped_input.clone();
    }

    std::vector<cv::Point3f> allPoints;
    for (const auto& pt : plateResult.final_plate1) {
        allPoints.push_back(cv::Point3f(pt.x, pt.y, pt.label));
    }
    for (const auto& pt : plateResult.final_plate2) {
        allPoints.push_back(cv::Point3f(pt.x, pt.y, pt.label));
    }
    for (const auto& pt : plateResult.icpPlatFid) {
        allPoints.push_back(cv::Point3f(pt.x, pt.y, pt.label));
    }

    for (const auto& pt : allPoints) {
        cv::Point2f center(pt.x, pt.y);
        std::string label = std::to_string(static_cast<int>(pt.z));
        
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
        cv::Point2f textPos = center - cv::Point2f(textSize.width/2.0f, -textSize.height/2.0f);
        
        cv::putText(img_color, label, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }

    std::string indexed_output = output_dir + "/indexed_fiducials.png";
    cv::imwrite(indexed_output, img_color);

    ofstream outFile(output_dir + "/output_" + position + ".json");
    outFile << setw(4) << inputJson << endl;

    cout << "Processing complete." << endl;
    return 0;
}