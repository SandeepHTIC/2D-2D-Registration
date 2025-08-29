#include "Center_detection.h"
#include "DBSCAN.h"
#include "Outlier_angle.h"

#include <algorithm>
#include <cmath>

using namespace std;
using namespace cv;

pair<Point2f, vector<Point2f>> Center_detection(
    const vector<Point2f>& centroid, int r, int c) {

    Point2f C(0, 0);
    vector<Point2f> first_ring_balls;

    if (centroid.empty()) return { C, first_ring_balls };

    // === Compute distances from image center ===
    Point2f center_img(c / 2.0f, r / 2.0f);
    vector<pair<float, int>> D;
    for (int i = 0; i < centroid.size(); ++i) {
        float dist = norm(center_img - centroid[i]);
        D.emplace_back(dist, i);
    }

    // Sort based on distance to image center
    sort(D.begin(), D.end());

    // Select 5 closest points to center
    vector<int> index;
    for (int i = 0; i < min(5, (int)D.size()); ++i) {
        index.push_back(D[i].second);
    }

    // Extract candidate center points
    vector<Point2f> centre_range;
    for (int idx : index) {
        centre_range.push_back(centroid[idx]);
    }

    try {
        for (int i = 0; i < centre_range.size(); ++i) {
            // Build new centroid list excluding the i-th candidate center
            vector<Point2f> new_centroid;
            for (int j = 0; j < centroid.size(); ++j) {
                if (centroid[j] != centre_range[i]) {
                    new_centroid.push_back(centroid[j]);
                }
            }

            // Compute distances to this candidate
            vector<pair<float, int>> r1;
            for (int j = 0; j < new_centroid.size(); ++j) {
                float dist = norm(centre_range[i] - new_centroid[j]);
                r1.emplace_back(dist, j);
            }

            // Sort and pick 20 nearest neighbors
            sort(r1.begin(), r1.end());
            vector<double> dist_vec;
            vector<int> new_ind;
            for (int k = 0; k < min(20, (int)r1.size()); ++k) {
                dist_vec.push_back(r1[k].first);
                new_ind.push_back(r1[k].second);
            }

            // Apply DBSCAN
            vector<int> idx = DBSCAN(dist_vec, 1.0, 3);

            // Count cluster 1 members
            int count = count_if(idx.begin(), idx.end(), [](int label) { return label == 1; });

            if (count >= 4) {
                C = centre_range[i];

                // Extract cluster 1 points
                vector<Point2f> selected;
                for (int m = 0; m < idx.size(); ++m) {
                    if (idx[m] == 1) {
                        selected.push_back(new_centroid[new_ind[m]]);
                    }
                }

                // Angle-based filtering
                first_ring_balls = Outlier_angle(selected, C);
                // Inside Center_detection(), before return
                std::cout << "[Center_detection] Detected Center: (" << C.x << ", " << C.y << ")" << std::endl;

                return { C, first_ring_balls };
            }
        }
    } catch (...) {
        C = Point2f(0, 0);
        first_ring_balls.clear();
    }

    return { C, first_ring_balls };
}