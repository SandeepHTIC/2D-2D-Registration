
#include "Outlier_angle.h"
#include "S_F_B.h"

#include <cmath>
#include <numeric>
#include <algorithm>

using namespace std;
using namespace cv;

std::vector<Point2f> Outlier_angle(const std::vector<Point2f>& FRB, const Point2f& Centre) {
    vector<Point2f> T_R_B;

    for (int k = 0; k < static_cast<int>(FRB.size()); ++k) {
        Point2f V1 = FRB[k] - Centre;
        float normV1 = norm(V1);
        if (normV1 < 1e-6) continue;  // Avoid division by zero
        V1 /= normV1;

        // Copy and remove k-th point
        vector<Point2f> F_R = FRB;
        F_R.erase(F_R.begin() + k);

        vector<float> angA;
        for (const auto& ball2 : F_R) {
            Point2f V2 = ball2 - Centre;
            float normV2 = norm(V2);
            if (normV2 < 1e-6) continue;
            V2 /= normV2;

            float dotProd = V1.dot(V2);
            dotProd = std::max(-1.0f, std::min(1.0f, dotProd));  // Clamp for safety
            float angleDeg = acos(dotProd) * 180.0f / CV_PI;
            angA.push_back(angleDeg);
        }

        // Apply angle offset and modulo 45
        for (auto& a : angA) {
            a = fmod(a + 2.0f, 45.0f);
        }

        // Count how many are within 3 degrees of a 45° multiple
        int count = std::count_if(angA.begin(), angA.end(), [](float a) {
            return a < 3.0f;
        });

        if (count >= 3) {
            T_R_B.push_back(FRB[k]);
        }
    }

    // Fallback using S_F_B if not enough points
    if (T_R_B.size() < 5) {
        if (!T_R_B.empty()) {
            T_R_B = S_F_B(T_R_B[0], Centre);
        } else {
            T_R_B.clear();  // Or leave it empty
        }
    }

    return T_R_B;
}
