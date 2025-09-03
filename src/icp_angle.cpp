#include "icp_angle.h"
#include <cmath>

#include <algorithm>
#include <numeric>

using namespace std;
using namespace cv;

static float pdist2(const Point2f &a, const Point2f &b)
{
    return norm(a - b);
}

ICPAngleResult icp_angle(
    const vector<Point2f> &first_ring_balls,
    const Point2f &Centre,
    const vector<vector<float>> &icpFid,
    vector<vector<float>> &add_CMM_Pts)
{
    ICPAngleResult result;
    if (icpFid.empty())
    {
        result.fRPts = first_ring_balls;
        result.fRPtsRem = {Point2f(0, 0)};
        result.icpFidFin = {};
        return result;
    }

    // 1. Compute icpCenterDist and prepend to icpFid
    vector<vector<float>> icpFidWithDist = icpFid;
    for (size_t i = 0; i < icpFidWithDist.size(); ++i)
    {
        float d = pdist2(Point2f(icpFidWithDist[i][1], icpFidWithDist[i][2]), Centre);
        icpFidWithDist[i].insert(icpFidWithDist[i].begin(), d);
    }
    // 2. Sort by distance
    sort(icpFidWithDist.begin(), icpFidWithDist.end(),
         [](const vector<float> &a, const vector<float> &b)
         { return a[0] < b[0]; });

    // 3. Compute icpSelAng matrix
    size_t N = icpFidWithDist.size();
    vector<vector<float>> icpSelAng(N, vector<float>(N, 0));
    for (size_t i = 0; i < N; ++i)
    {
        Point2f point0(icpFidWithDist[i][2], icpFidWithDist[i][3]);
        for (size_t j = 0; j < N; ++j)
        {
            Point2f point1(icpFidWithDist[j][2], icpFidWithDist[j][3]);
            Point2f n1 = (point0 - Centre) * (1.0f / norm(point0 - Centre));
            Point2f n2 = (point1 - Centre) * (1.0f / norm(point1 - Centre));
            float detn = n2.x * n1.y - n2.y * n1.x;
            float dotn = n1.dot(n2);
            icpSelAng[j][i] = atan2f(fabs(detn), dotn) * 180.0f / CV_PI;
        }
    }

    // 4. Find icpFidFin (simple version: pick the first row for demonstration)
    vector<float> icpFidFin;
    if (!icpFidWithDist.empty())
    {
        icpFidFin = icpFidWithDist[0];
    }
    result.icpFidFin = icpFidFin;

    // 5. Compute icpAng for first_ring_balls
    vector<float> icpAng;
    Point2f point1(icpFidFin.size() > 3 ? icpFidFin[2] : 0, icpFidFin.size() > 4 ? icpFidFin[3] : 0);
    for (const auto &pt : first_ring_balls)
    {
        Point2f n1 = (pt - Centre) * (1.0f / norm(pt - Centre));
        Point2f n2 = (point1 - Centre) * (1.0f / norm(point1 - Centre));
        float detn = n2.x * n1.y - n2.y * n1.x;
        float dotn = n1.dot(n2);
        icpAng.push_back(atan2f(fabs(detn), dotn) * 180.0f / CV_PI);
    }

    // 6. Sort first_ring_balls by icpAng
    vector<size_t> idx(icpAng.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&](size_t i1, size_t i2)
         { return icpAng[i1] < icpAng[i2]; });
    vector<Point2f> sorted_ring;
    vector<float> sorted_icpAng;
    for (auto i : idx)
    {
        sorted_ring.push_back(first_ring_balls[i]);
        sorted_icpAng.push_back(icpAng[i]);
    }
    result.fRPts = sorted_ring;
    // For demonstration, fRPtsRem is empty
    result.fRPtsRem = {};
    return result;
}
