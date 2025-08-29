#include "S_F_B.h"
#include <cmath>

using namespace std;
using namespace cv;

std::vector<Point2f> S_F_B(const Point2f& ball, const Point2f& C) {
    vector<Point2f> FRB;
    float x = 45.0f;

    // Vector from center to input point
    Point2f V = ball - C;

    for (int i = 1; i <= 8; ++i) {
        float angle_deg = x * i;
        float angle_rad = angle_deg * static_cast<float>(CV_PI) / 180.0f;

        // 2D rotation matrix manually
        float cosA = cos(angle_rad);
        float sinA = sin(angle_rad);
        Point2f rotated(
            cosA * V.x - sinA * V.y,
            sinA * V.x + cosA * V.y
        );

        // Add rotated point to center
        Point2f p = C + rotated;
        FRB.push_back(p);
    }

    return FRB;
}