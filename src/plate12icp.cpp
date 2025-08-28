#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <set>
#include <map>
#include "plate12icp.h"
#include "kmeans.h"
#include "DBSCAN.h"

using namespace cv;
using namespace std;

// Helper to compute angle between two points w.r.t. center
static float angleBetween(const Point2f& a, const Point2f& b) {
    return atan2f(b.y - a.y, b.x - a.x);
}

PlateFiducials plate12icp(const cv::Mat& blob_image,
                          const cv::Point2f& Centre,
                          const std::vector<cv::Point2f>& icpFid)
{
    PlateFiducials output;
    vector<Point2f> allCentroids;
    vector<float> allAreas;

    vector<vector<Point>> contours;
    findContours(blob_image.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (auto& cnt : contours) {
        double area = contourArea(cnt);
        if (area >= 5.0) {
            Moments m = moments(cnt);
            if (m.m00 != 0) {
                Point2f c(m.m10 / m.m00, m.m01 / m.m00);
                allCentroids.push_back(c);
                allAreas.push_back(static_cast<float>(area));
            }
        }
    }

    int N = static_cast<int>(allCentroids.size());
    if (N == 0) return output;

    // === STEP 1: Outer ring via radial KMeans ===
    vector<float> radialDistances;
    for (const auto& pt : allCentroids)
        radialDistances.push_back(norm(pt - Centre));

    int Kdist = (N >= 50) ? 4 : 3;
    Mat distMat(N, 1, CV_32F);
    for (int i = 0; i < N; ++i)
        distMat.at<float>(i, 0) = radialDistances[i];

    Mat labelsDist, centersDist;
    kmeans(distMat, Kdist, labelsDist,
           TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 100, 1.0),
           3, KMEANS_PP_CENTERS, centersDist);

    vector<pair<float, int>> dist_cluster;
    for (int i = 0; i < Kdist; ++i)
        dist_cluster.emplace_back(centersDist.at<float>(i), i);
    sort(dist_cluster.begin(), dist_cluster.end());

    int outerLabel = dist_cluster.back().second;

    vector<Point2f> outerCentroids;
    vector<float> outerAreas;
    vector<float> outerRadii;
    for (int i = 0; i < N; ++i) {
        if (labelsDist.at<int>(i) == outerLabel) {
            outerCentroids.push_back(allCentroids[i]);
            outerAreas.push_back(allAreas[i]);
            outerRadii.push_back(norm(allCentroids[i] - Centre));
        }
    }

    cout << "[INFO] Outer ring size (KMeans): " << outerCentroids.size() << endl;

    // === STEP 2: Area-based KMeans ===
    int Karea = 3;
    Mat areaMat((int)outerAreas.size(), 1, CV_32F);
    for (int i = 0; i < areaMat.rows; ++i)
        areaMat.at<float>(i, 0) = outerAreas[i];

    Mat areaLabels, areaCenters;
    kmeans(areaMat, Karea, areaLabels,
           TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 100, 1.0),
           3, KMEANS_PP_CENTERS, areaCenters);

    vector<pair<float, int>> centerSorted;
    for (int i = 0; i < Karea; ++i)
        centerSorted.emplace_back(areaCenters.at<float>(i), i);
    sort(centerSorted.begin(), centerSorted.end());

    int midLabel = centerSorted[1].second;
    int largeLabel = centerSorted[2].second;

    vector<Point2f> plate1, plate2_raw;

    for (int i = 0; i < areaLabels.rows; ++i) {
        if (areaLabels.at<int>(i) == midLabel)
            plate1.push_back(outerCentroids[i]);
        else if (areaLabels.at<int>(i) == largeLabel)
            plate2_raw.push_back(outerCentroids[i]);
    }

    // === STEP 3: Angular filtering (fallback if DBSCAN fails) ===
    vector<float> angles;
    for (const auto& pt : plate2_raw)
        angles.push_back(angleBetween(Centre, pt));

    vector<double> angles_double(angles.begin(), angles.end());
    vector<int> labels = DBSCAN(angles_double, 5.0, 3); // 
    map<int, vector<Point2f>> clusters;

    for (size_t i = 0; i < plate2_raw.size(); ++i) {
        if (labels[i] != -1)
            clusters[labels[i]].push_back(plate2_raw[i]);
    }

    vector<Point2f> plate2;
    if (!clusters.empty()) {
        // Use largest angular cluster
        int bestLabel = -1;
        size_t maxSize = 0;
        for (auto& [label, group] : clusters) {
            if (group.size() > maxSize) {
                maxSize = group.size();
                bestLabel = label;
            }
        }
        plate2 = clusters[bestLabel];
        cout << "[INFO] Plate2 selected from DBSCAN cluster " << bestLabel
             << " with size: " << plate2.size() << endl;
    } else {
        // Fallback: use all and filter angular outliers
        float meanAngle = accumulate(angles.begin(), angles.end(), 0.0f) / angles.size();
        vector<pair<float, int>> angleDiff;
        for (int i = 0; i < (int)angles.size(); ++i)
            angleDiff.emplace_back(abs(angles[i] - meanAngle), i);

        sort(angleDiff.begin(), angleDiff.end());

        for (int i = 0; i < min(12, (int)angleDiff.size()); ++i)
            plate2.push_back(plate2_raw[angleDiff[i].second]);

        cout << "[INFO] Plate2 selected from angular fallback. Size: " << plate2.size() << endl;
    }

    // === STEP 4: Final filtering to move misclassified Plate 1 points ===
vector<Point2f> final_plate1;
vector<Point2f> final_plate2 = plate2;
set<int> plate1_indices_to_move;

const float ANGLE_STEP_DEG = 45.0f;
const float ANGLE_TOLERANCE_DEG = 5.0f; // +/- tolerance

auto angularDiffDeg = [&](const Point2f& a, const Point2f& b) {
    float ang1 = angleBetween(Centre, a);
    float ang2 = angleBetween(Centre, b);
    float diff = fabs(ang1 - ang2);
    if (diff > CV_PI) diff = 2 * CV_PI - diff;
    return diff * 180.0f / CV_PI;
};

auto isMultipleOf45 = [&](float diffDeg) {
    float modVal = fmod(diffDeg, ANGLE_STEP_DEG);
    // Check closeness to 0 or to 45 within tolerance
    return (modVal <= ANGLE_TOLERANCE_DEG || 
            fabs(ANGLE_STEP_DEG - modVal) <= ANGLE_TOLERANCE_DEG);
};

// Condition: Check if a plate1 point is at ~multiple of 45° from any plate2 point
for (size_t i = 0; i < plate1.size(); ++i) {
    for (const auto& p2 : final_plate2) {
        float diffDeg = angularDiffDeg(plate1[i], p2);
        if (isMultipleOf45(diffDeg)) {
            plate1_indices_to_move.insert(i);
            break;
        }
    }
}

// Perform the move for all marked points
if (!plate1_indices_to_move.empty()) {
    cout << "[INFO] Moving " << plate1_indices_to_move.size() 
         << " points from Plate1 to Plate2." << endl;
    for (size_t i = 0; i < plate1.size(); ++i) {
        if (plate1_indices_to_move.count(i)) {
            final_plate2.push_back(plate1[i]);
        } else {
            final_plate1.push_back(plate1[i]);
        }
    }
} else {
    final_plate1 = plate1;
}

cout << "[INFO] Final Plate1 size: " << final_plate1.size() << endl;
cout << "[INFO] Final Plate2 size: " << final_plate2.size() << endl;

// Convert Point2f vectors to LabeledPoint vectors
output.final_plate1.clear();
output.final_plate2.clear();

// Convert final_plate1 to LabeledPoint (without labels for now)
for (const auto& pt : final_plate1) {
    output.final_plate1.push_back({pt.x, pt.y, 0}); // label 0 for unlabeled
}

// Convert final_plate2 to LabeledPoint (without labels for now)  
for (const auto& pt : final_plate2) {
    output.final_plate2.push_back({pt.x, pt.y, 0}); // label 0 for unlabeled
}

return output;

}