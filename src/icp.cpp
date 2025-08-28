#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <fstream>
#include <algorithm>
#include "kmeans.h"
#include "dbscan.h"
#include "utils.h"

using namespace cv;
using namespace std;

cv::Mat plate12withICP_noInputs(const cv::Mat& blob_image, const cv::Point2f& Centre, const std::string& path_d) {
    // Step 1: Extract all centroids from binary blob image
    vector<Point2f> centroids;
    Mat labels, stats, centroidsMat;
    int n_labels = connectedComponentsWithStats(blob_image, labels, stats, centroidsMat);

    for (int i = 1; i < n_labels; ++i) {
        Point2f pt(centroidsMat.at<double>(i, 0), centroidsMat.at<double>(i, 1));
        centroids.push_back(pt);
    }

    int N = centroids.size();
    if (N < 20) {
        cerr << "Too few points to proceed." << endl;
        return blob_image;
    }

    // Step 2: Compute radius of each point w.r.t. Centre
    vector<float> radii(N);
    for (int i = 0; i < N; ++i) {
        radii[i] = norm(centroids[i] - Centre);
    }

    // Step 3: KMeans clustering on radius to get 2 clusters (plate1 and plate2)
    vector<vector<float>> radius_data(N, vector<float>(1));
    for (int i = 0; i < N; ++i) radius_data[i][0] = radii[i];

    vector<int> cluster_labels;
    kmeans(radius_data, 2, cluster_labels);

    // Step 4: Assign clusters based on mean radius (smaller radius = plate1)
    vector<float> mean_r(2, 0.0f);
    vector<int> count_r(2, 0);
    for (int i = 0; i < N; ++i) {
        mean_r[cluster_labels[i]] += radii[i];
        count_r[cluster_labels[i]]++;
    }
    mean_r[0] /= count_r[0];
    mean_r[1] /= count_r[1];

    int plate1_id = mean_r[0] < mean_r[1] ? 0 : 1;
    int plate2_id = 1 - plate1_id;

    vector<Point2f> plate1_pts, plate2_pts;
    for (int i = 0; i < N; ++i) {
        if (cluster_labels[i] == plate1_id) plate1_pts.push_back(centroids[i]);
        else plate2_pts.push_back(centroids[i]);
    }

    // Step 5: DBSCAN on plate2 to remove outliers
    vector<Point2f> plate2_dbscan_pts;
    if (!plate2_pts.empty()) {
        vector<vector<float>> db_pts;
        for (const auto& pt : plate2_pts)
            db_pts.push_back({pt.x, pt.y});

        vector<int> db_labels;
        DBSCAN dbscan(20.0f, 4);
        dbscan.fit(db_pts, db_labels);

        map<int, vector<Point2f>> clusters;
        for (int i = 0; i < db_labels.size(); ++i) {
            if (db_labels[i] != -1) clusters[db_labels[i]].push_back(plate2_pts[i]);
        }

        if (!clusters.empty()) {
            int max_id = max_element(clusters.begin(), clusters.end(), [](auto& a, auto& b) {
                return a.second.size() < b.second.size();
            })->first;
            plate2_dbscan_pts = clusters[max_id];
        } else {
            plate2_dbscan_pts = plate2_pts;
        }
    }

    // Step 6: Angular filtering fallback (mod 45 deg)
    if (plate2_dbscan_pts.size() < 12) {
        plate2_dbscan_pts.clear();
        vector<pair<float, Point2f>> angle_pts;
        for (const auto& pt : plate2_pts) {
            float angle = atan2(pt.y - Centre.y, pt.x - Centre.x) * 180.0f / CV_PI;
            if (angle < 0) angle += 360.0f;
            float a = fmod(angle, 45.0f);
            if (a < 5.0f || a > 40.0f) angle_pts.push_back({angle, pt});
        }
        sort(angle_pts.begin(), angle_pts.end());
        for (const auto& ap : angle_pts) plate2_dbscan_pts.push_back(ap.second);
    }

    // Step 7: Distance filtering (within 40 units of max)
    vector<float> dist_plate2;
    for (auto& pt : plate2_dbscan_pts) dist_plate2.push_back(norm(pt - Centre));
    float max_dist = *max_element(dist_plate2.begin(), dist_plate2.end());
    vector<Point2f> plate2_final;
    for (int i = 0; i < plate2_dbscan_pts.size(); ++i) {
        if (abs(dist_plate2[i] - max_dist) < 40.0f)
            plate2_final.push_back(plate2_dbscan_pts[i]);
    }

    // Step 8: Outlier removal on plate1 (15%-100% percentile range)
    vector<float> dist1;
    for (auto& pt : plate1_pts) dist1.push_back(norm(pt - Centre));
    sort(dist1.begin(), dist1.end());
    float low = dist1[dist1.size() * 0.15f];
    float high = dist1[dist1.size() * 1.0f - 1];

    vector<Point2f> plate1_final;
    for (int i = 0; i < plate1_pts.size(); ++i) {
        float d = norm(plate1_pts[i] - Centre);
        if (d >= low && d <= high) plate1_final.push_back(plate1_pts[i]);
    }

    // Step 9: Draw results
    cv::Mat outputImg;
    cv::cvtColor(blob_image, outputImg, COLOR_GRAY2BGR);

    for (const auto& pt : plate1_final)
        cv::circle(outputImg, pt, 3, Scalar(0, 255, 0), -1); // Green

    for (const auto& pt : plate2_final)
        cv::circle(outputImg, pt, 3, Scalar(0, 0, 255), -1); // Red

    // Step 10: Save image
    string outPath = path_d + "/plate_fiducials_result.png";
    imwrite(outPath, outputImg);

    return outputImg;
}