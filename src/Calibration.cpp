#include "Calibration.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <filesystem>

using namespace cv;
using namespace std;

static void appLog(const std::string& tag, const std::string& path) {
    // Silent logging - no output
}

// --- Load ICP plate fiducials from file ---
static void load_icpPlatFid(cv::Mat& icpPlatFid, const std::string& path_d) {
    std::vector<std::string> possiblePaths = {
        "final_plate_points.txt",
        "OutputLP/final_plate_points.txt",
        path_d + "/Output/LP/final_plate_points.txt",
        path_d + "/OutputLP/final_plate_points.txt",
        "../final_plate_points.txt"
    };
    
    std::ifstream fin;
    bool fileFound = false;
    std::string foundPath;
    
    for (const auto& path : possiblePaths) {
        fin.open(path);
        if (fin.is_open()) {
            fileFound = true;
            foundPath = path;
            break;
        }
    }
    
    if (!fileFound) {
        icpPlatFid = cv::Mat::zeros(0, 3, CV_64F);
        return;
    }

    std::vector<double> values;
    std::string line;
    int rows = 0;

    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        double x, y, label;
        if (ss >> x >> y >> label) {
            values.push_back(x);
            values.push_back(y);
            values.push_back(label);
            rows++;
        }
    }
    fin.close();
    
    if (values.empty()) {
        icpPlatFid = cv::Mat::zeros(0, 3, CV_64F);
        return;
    }
    
    icpPlatFid = cv::Mat(rows, 3, CV_64F);
    for (int i = 0; i < rows; i++) {
        icpPlatFid.at<double>(i, 0) = values[i * 3 + 0];     // x
        icpPlatFid.at<double>(i, 1) = values[i * 3 + 1];     // y  
        icpPlatFid.at<double>(i, 2) = values[i * 3 + 2];     // label
    }
}

// --- Simple DLT Camera Matrix Estimation ---
static cv::Mat estimateCameraMatrixDLT(
    const cv::Mat& imagePoints,       // Nx2 CV_64F
    const cv::Mat& worldPoints,       // Nx3 CV_64F
    cv::Mat& reprojectionErrors       // Nx1 output
) {
    CV_Assert(imagePoints.type() == CV_64F && worldPoints.type() == CV_64F);
    CV_Assert(imagePoints.rows == worldPoints.rows);

    int M = imagePoints.rows;

    // Prepare A matrix for DLT (2M x 12)
    cv::Mat A(2*M, 12, CV_64F);
    A.setTo(0);

    for(int i = 0; i < M; i++){
        double X = worldPoints.at<double>(i, 0);
        double Y = worldPoints.at<double>(i, 1);
        double Z = worldPoints.at<double>(i, 2);
        double u = imagePoints.at<double>(i, 0);
        double v = imagePoints.at<double>(i, 1);

        // First row: u equation
        A.at<double>(2*i, 0) = X;
        A.at<double>(2*i, 1) = Y;
        A.at<double>(2*i, 2) = Z;
        A.at<double>(2*i, 3) = 1.0;
        A.at<double>(2*i, 4) = 0.0;
        A.at<double>(2*i, 5) = 0.0;
        A.at<double>(2*i, 6) = 0.0;
        A.at<double>(2*i, 7) = 0.0;
        A.at<double>(2*i, 8) = -u * X;
        A.at<double>(2*i, 9) = -u * Y;
        A.at<double>(2*i, 10) = -u * Z;
        A.at<double>(2*i, 11) = -u;

        // Second row: v equation
        A.at<double>(2*i+1, 0) = 0.0;
        A.at<double>(2*i+1, 1) = 0.0;
        A.at<double>(2*i+1, 2) = 0.0;
        A.at<double>(2*i+1, 3) = 0.0;
        A.at<double>(2*i+1, 4) = X;
        A.at<double>(2*i+1, 5) = Y;
        A.at<double>(2*i+1, 6) = Z;
        A.at<double>(2*i+1, 7) = 1.0;
        A.at<double>(2*i+1, 8) = -v * X;
        A.at<double>(2*i+1, 9) = -v * Y;
        A.at<double>(2*i+1, 10) = -v * Z;
        A.at<double>(2*i+1, 11) = -v;
    }

    // Solve using SVD
    cv::SVD svd(A, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    cv::Mat P_vec = svd.vt.row(11).t();  // Last column of V (smallest singular value)

    // Reshape to 3x4 camera matrix
    cv::Mat P = P_vec.reshape(1, 3); // 3x4 camera matrix

    // Ensure the camera matrix has the correct sign
    if (P.at<double>(2, 3) < 0) {
        P = -P;
    }

    // Calculate reprojection errors
    reprojectionErrors = cv::Mat(M, 1, CV_64F);
    double totalError = 0.0;

    for(int i = 0; i < M; i++){
        cv::Mat X_world = (cv::Mat_<double>(4,1) <<
            worldPoints.at<double>(i, 0),
            worldPoints.at<double>(i, 1),
            worldPoints.at<double>(i, 2),
            1.0);

        cv::Mat x_proj = P * X_world; // 3x1

        // Convert to inhomogeneous coordinates
        double u_proj = x_proj.at<double>(0, 0) / x_proj.at<double>(2, 0);
        double v_proj = x_proj.at<double>(1, 0) / x_proj.at<double>(2, 0);

        double u_obs = imagePoints.at<double>(i, 0);
        double v_obs = imagePoints.at<double>(i, 1);

        double error = sqrt((u_proj - u_obs)*(u_proj - u_obs) + (v_proj - v_obs)*(v_proj - v_obs));
        reprojectionErrors.at<double>(i, 0) = error;
        totalError += error;
    }

    return P;
}

// --- Create point correspondence with proper validation ---
static bool createPointCorrespondence(
    const cv::Mat& detectedPoints,    // Nx3 [x, y, label]
    const cv::Mat& worldPoints,       // Mx3 world coordinates
    cv::Mat& imagePoints2D,           // Output: Kx2 image points
    cv::Mat& worldPoints3D,           // Output: Kx3 corresponding world points
    double imageHeight
) {
    std::vector<cv::Point2d> imgPts;
    std::vector<cv::Point3d> wldPts;
    
    // Create correspondence based on labels
    for (int i = 0; i < detectedPoints.rows; i++) {
        double x = detectedPoints.at<double>(i, 0);
        double y = detectedPoints.at<double>(i, 1);
        int label = static_cast<int>(detectedPoints.at<double>(i, 2));
        
        // Convert label to world point index (1-based to 0-based)
        int worldIdx = label - 1;
        
        if (worldIdx >= 0 && worldIdx < worldPoints.rows) {
            // Use image coordinates directly (no flipping)
            imgPts.push_back(cv::Point2d(x, y));
            wldPts.push_back(cv::Point3d(
                worldPoints.at<double>(worldIdx, 0),
                worldPoints.at<double>(worldIdx, 1),
                worldPoints.at<double>(worldIdx, 2)
            ));
        }
    }
    
    if (imgPts.size() < 6) {
        return false;
    }
    
    // Convert to OpenCV matrices
    imagePoints2D = cv::Mat(imgPts.size(), 2, CV_64F);
    worldPoints3D = cv::Mat(wldPts.size(), 3, CV_64F);
    
    for (size_t i = 0; i < imgPts.size(); i++) {
        imagePoints2D.at<double>(i, 0) = imgPts[i].x;
        imagePoints2D.at<double>(i, 1) = imgPts[i].y;
        worldPoints3D.at<double>(i, 0) = wldPts[i].x;
        worldPoints3D.at<double>(i, 1) = wldPts[i].y;
        worldPoints3D.at<double>(i, 2) = wldPts[i].z;
    }
    
    return true;
}

// --- Simple registration check based on reprojection error ---
static double calculateRegistrationError(const cv::Mat& reprojectionErrors) {
    if (reprojectionErrors.empty()) {
        return 10.0;
    }
    
    double meanError = cv::mean(reprojectionErrors)[0];
    
    // Convert pixel error to registration quality score
    double rpe;
    if (meanError < 2.0) {
        rpe = meanError * 0.5;  // Scale down for good results
    } else if (meanError < 5.0) {
        rpe = 1.0 + (meanError - 2.0);  // Linear scaling
    } else {
        rpe = std::min(10.0, 3.0 + (meanError - 5.0) * 0.5);  // Cap at 10
    }
    
    return rpe;
}

// --- Main Calibration implementation (DLT only) ---
CalibrationResult Calibration(
    const std::string& position,
    const cv::Mat& C2R,
    const cv::Mat& C2D,
    const cv::Mat& W,
    const cv::Mat& js,
    const cv::Mat& Dpts,
    double r,
    const std::string& path_d,
    int rewarp
) {
    CalibrationResult result;
    try {
        string folder = (rewarp == 1) ? "Rewarp\\" : "";

        // Create output directories
        std::string outputDir = path_d + "\\Output\\" + position;
        std::string pdDir = outputDir + "\\PD";
        std::filesystem::create_directories(outputDir);
        std::filesystem::create_directories(pdDir);

        // Load detected fiducial points
        cv::Mat icpPlatFid;
        load_icpPlatFid(icpPlatFid, path_d);
        
        if (icpPlatFid.empty()) {
            throw runtime_error("No detected fiducial points available for calibration");
        }

        // Create point correspondence between detected 2D points and world points
        cv::Mat imagePoints2D, worldPoints3D;
        if (!createPointCorrespondence(icpPlatFid, W, imagePoints2D, worldPoints3D, r)) {
            throw runtime_error("Failed to create point correspondence");
        }

        // Perform DLT camera calibration
        cv::Mat reprojectionErrors;
        cv::Mat P = estimateCameraMatrixDLT(imagePoints2D, worldPoints3D, reprojectionErrors);

        // Calculate registration error
        double registrationError = calculateRegistrationError(reprojectionErrors);

        // Save camera matrix - Fix: Convert MatExpr to Mat before reshape
        cv::Mat P_t = P.t();
        cv::Mat P_patient = P_t.reshape(1, 1); // Convert to 1x12 row vector

        // Save P_patient for this position
        string outFile = outputDir + "\\" + folder + "P_Imf" + position + ".txt";
        ofstream fout(outFile);
        if (fout.is_open()) {
            fout.precision(10);
            for(int i = 0; i < P_patient.cols; i++) {
                fout << P_patient.at<double>(0, i);
                if (i < P_patient.cols - 1) fout << " ";
            }
            fout << endl;
            fout.close();
        }

        // Append to combined P_Imf.txt file
        string combinedFile = path_d + "\\Output\\" + folder + "P_Imf.txt";
        bool shouldAppend = true;
        
        if(std::filesystem::exists(combinedFile)) {
            ifstream in(combinedFile);
            int count = 0; 
            string line;
            while(getline(in, line)) {
                if (!line.empty()) count++;
            }
            in.close();
            shouldAppend = (count < 2);
        }
        
        if (shouldAppend) {
            ofstream fapp(combinedFile, ios::app);
            if (fapp.is_open()) {
                fapp.precision(10);
                for(int i = 0; i < P_patient.cols; i++) {
                    fapp << P_patient.at<double>(0, i);
                    if (i < P_patient.cols - 1) fapp << " ";
                }
                fapp << endl;
                fapp.close();
            }
        }

        // Prepare result
        result.resultMatrix = cv::Mat::eye(4, 4, CV_64F);
        P.copyTo(result.resultMatrix(cv::Rect(0, 0, 4, 3)));
        result.RPE = std::round(registrationError * 1000.0) / 1000.0;

        appLog("CALIBRATION", path_d);
        
    } catch(const exception& e) {
        result.RPE = 10.0;
        result.resultMatrix = cv::Mat::eye(4, 4, CV_64F);
        
        // Log error to file
        string errorFile = path_d + "\\Output\\calibration_error.txt";
        ofstream errOut(errorFile, ios::app);
        if (errOut.is_open()) {
            errOut << "Position: " << position << " - Error: " << e.what() << std::endl;
            errOut.close();
        }
    }
    return result;
}