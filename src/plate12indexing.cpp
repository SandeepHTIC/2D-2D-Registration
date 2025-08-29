// icp_simple.cpp
// Streamlined ICP-based indexing for two pre-refined point sets.
// No large 2D allocations; O(M+N) memory. Ready to paste.

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <limits>

#include "plate12icp.h"        // brings in LabeledPoint & PlateFiducials
#include "plate12indexing.h"   // if you need other pipeline declarations

using namespace cv;
using namespace std;

// ---------- Safety Helpers ----------
template<typename T, typename N>
inline void safe_reserve(vector<T>& v, N n) {
    long long nn = static_cast<long long>(n);
    if (nn > 0) {
        size_t cap = static_cast<size_t>(nn);
        if (cap > v.max_size()) cap = v.max_size();
        v.reserve(cap);
    }
}

// ---------- Geometry Helpers ----------
inline double rad2deg(double r) { return r * 180.0 / CV_PI; }
inline double atan2d(double y, double x) { return rad2deg(atan2(y, x)); }
inline double dot2d(const Point2d &a, const Point2d &b) { return a.x*b.x + a.y*b.y; }
inline double norm2d(const Point2d &p) { return sqrt(p.x*p.x + p.y*p.y); }
inline double det2d(const Point2d &a, const Point2d &b) { return a.x*b.y - a.y*b.x; }
inline Point2d normalizeVec(const Point2d& p) {
    double n = norm2d(p);
    return (n>0) ? Point2d(p.x/n, p.y/n) : Point2d(0,0);
}
inline double modpos(double x, double m) {
    double r = fmod(x, m);
    return (r<0) ? (r+m) : r;
}

// compute convex hull indices
static vector<int> convexHullIdx(const vector<Point2d>& pts) {
    vector<Point2f> ptsf;
    safe_reserve(ptsf, pts.size());
    for (auto &p : pts) ptsf.emplace_back((float)p.x,(float)p.y);
    vector<int> idx;
    if (!ptsf.empty()) convexHull(ptsf, idx, false, false);
    return idx;
}

// find for each vals[i] the index of closest element in ref[]
static vector<int> nearestIdx(const vector<double>& vals, const vector<double>& ref) {
    vector<int> out;
    safe_reserve(out, vals.size());
    for (double v: vals) {
        double best=1e12; int bi=0;
        for (int j=0;j<(int)ref.size();++j) {
            double d = fabs(v - ref[j]);
            if (d<best) { best=d; bi=j; }
        }
        out.push_back(bi);
    }
    return out;
}

// intersect by rounding or ceiling
static vector<int> intersectRounded(
    const vector<double>& a,
    const vector<int>& refSet,
    bool useRound
) {
    unordered_set<int> S(refSet.begin(), refSet.end());
    vector<int> out;
    safe_reserve(out, a.size());
    for (int i=0;i<(int)a.size();++i) {
        int v = useRound ? (int)llround(a[i]) : (int)ceil(a[i]);
        if (S.count(v)) out.push_back(i);
    }
    return out;
}

// ---------- Streamed Column Selection ----------
//
// For each Q[k], compute col[i] = angle(P[i],Q[k]) and find
// the minimal positive gap in sorted(col).  Return bestGaps[k].
// Then return bestK = argmax(bestGaps) and fill outCol = col(bestK).
static int selectBestColumn(
    const vector<Point2d>& P,
    const vector<Point2d>& Q,
    const Point2d& C,
    vector<double>& outCol,
    vector<double>& bestGaps
) {
    int M = (int)P.size(), N = (int)Q.size();
    bestGaps.assign(N, 0.0);
    if (M==0||N==0) {
        outCol.clear();
        return 0;
    }

    vector<double> col(M);
    for (int k=0;k<N;++k) {
        // build one column
        Point2d q = Q[k];
        for (int i=0;i<M;++i) {
            Point2d n1 = normalizeVec(P[i] - C);
            Point2d n2 = normalizeVec(q    - C);
            col[i] = atan2d(fabs(det2d(n2,n1)), dot2d(n1,n2));
        }
        sort(col.begin(), col.end());
        double best=0.0, mn=1e12;
        for (int i=1;i<M;++i) {
            double d = col[i] - col[i-1];
            if (d>0 && d<mn) { mn=d; best=d; }
        }
        bestGaps[k] = best;
    }

    int bestK = (int)(max_element(bestGaps.begin(),bestGaps.end()) - bestGaps.begin());

    // recompute outCol for bestK
    outCol.resize(M);
    Point2d qb = Q[bestK];
    for (int i=0;i<M;++i) {
        Point2d n1 = normalizeVec(P[i] - C);
        Point2d n2 = normalizeVec(qb  - C);
        outCol[i] = atan2d(fabs(det2d(n2,n1)), dot2d(n1,n2));
    }
    return bestK;
}

// ---------- Reference Matching & Refinement ----------

// Filter P/angles by matching to any column of refAngles (R×C)
// useRound: true → round(), false → ceil()
static void filterByReference(
    const vector<Point2d>& P,
    const vector<double>& ang,
    const vector<vector<double>>& refAngles,
    bool useRound,
    vector<Point2d>& outP,
    vector<double>& outAng
) {
    int R = (int)refAngles.size();
    int C = (int)refAngles[0].size();

    // Pair angles with original indices so sorting keeps points aligned
    vector<pair<double,int>> angWithIdx;
    angWithIdx.reserve(ang.size());
    for (int i=0;i<(int)ang.size();++i) angWithIdx.emplace_back(ang[i], i);
    sort(angWithIdx.begin(), angWithIdx.end(), [](const auto& a, const auto& b){return a.first < b.first;});

    // Build sorted angle list
    vector<double> sortedAng;
    sortedAng.reserve(angWithIdx.size());
    for (auto& p : angWithIdx) sortedAng.push_back(p.first);

    for (int c=0;c<C;++c) {
        vector<int> look(R);
        for (int r=0;r<R;++r) look[r] = (int)refAngles[r][c];
        auto posInSorted = intersectRounded(sortedAng, look, useRound); // positions in sortedAng
        if ((int)posInSorted.size() > R/2) {
            outP.clear(); outAng.clear();
            for (int pos : posInSorted) {
                int origIdx = angWithIdx[pos].second;
                outP.push_back(P[origIdx]);
                outAng.push_back(sortedAng[pos]);
            }
            return;
        }
    }
    // fallback: take all
    outP = P;
    outAng = ang;
}

// Compute RMS error to best-match each angle to refAngles(R×C)
static vector<double> computeRmsToRef(
    const vector<double>& vals,
    const vector<vector<double>>& refAngles
){
    int R = (int)refAngles.size();
    int C = (R > 0) ? (int)refAngles[0].size() : 0;
    int N = (int)vals.size();
    vector<double> rms(C,0.0);

    vector<double> s(vals);
    sort(s.begin(), s.end());

    for (int c=0;c<C;++c) {
        double S=0.0;
        for (double v : s) {
            double mn=1e12;
            for (int r=0;r<R;++r)
                mn = min(mn, fabs(v - refAngles[r][c]));
            S += mn*mn;
        }
        rms[c] = sqrt(S);
    }
    return rms;
}

// Drop extremes until minRms ≤5.0
static void refineByRms(
    vector<Point2d>& P,
    vector<double>& ang,
    const vector<vector<double>>& refAngles
) {
    int cnt=0;
    vector<Point2d> bP; vector<double> bA;
    while (true) {
        auto rms = computeRmsToRef(ang, refAngles);
        double mn = *min_element(rms.begin(), rms.end());
        if (mn <= 5.0) break;
        if (cnt==0 && ang.size()>1) {
            bP = P; bA = ang;
            ang.erase(ang.begin());
            P.erase(P.begin());
            if (!ang.empty()) {
                ang.pop_back();
                P.pop_back();
            }
        } else if (cnt>0 && cnt < (int)ang.size()) {
            P = bP; ang = bA;
            P.erase(P.begin()+cnt);
            ang.erase(ang.begin()+cnt);
        } else break;
        cnt++;
    }
}

// nearestpoint function equivalent to MATLAB's nearestpoint
static vector<int> nearestpoint(const vector<double>& vals, const vector<double>& ref) {
    vector<int> indices;
    safe_reserve(indices, vals.size());
    
    for (double val : vals) {
        double minDist = 1e12;
        int bestIdx = 0;
        for (int i=0; i<(int)ref.size(); ++i) {
            double dist = fabs(val - ref[i]);
            if (dist < minDist) {
                minDist = dist;
                bestIdx = i;
            }
        }
        indices.push_back(bestIdx);
    }
    return indices;
}

// ---------- Core ICP Indexing Streamed ----------

static vector<LabeledPoint> icp_stream(
    const vector<Point2d>& plate1Pts,
    const vector<Point2d>& plate2Pts,
    const vector<Vec3d>& icpRaw,
    const Point2d& Centre
) {
    // 1) hull-order input plates
    auto h1 = convexHullIdx(plate1Pts);
    vector<Point2d> p1; safe_reserve(p1, h1.size());
    for (int i : h1) p1.push_back(plate1Pts[i]);

    auto h2 = convexHullIdx(plate2Pts);
    vector<Point2d> p2; safe_reserve(p2, h2.size());
    for (int i : h2) p2.push_back(plate2Pts[i]);

    // 2) filter raw ICP ≥250 (matching MATLAB logic)
    vector<Vec3d> icpFid;
    for (auto &r : icpRaw) {
        if (r[0] >= 240.0) {  // Match MATLAB threshold exactly
            icpFid.push_back(r);
        }
    }
    if (icpFid.empty()) {
        return {};
    }

    // ===== Plate1 Indexing =====
    // Extract 2D points from icpFid for column selection
    vector<Point2d> icpPts;
    safe_reserve(icpPts, icpFid.size());
    for (auto &r : icpFid) {
        icpPts.emplace_back(r[1], r[2]);
    }
    
    vector<double> gaps1, ang1;
    selectBestColumn(p1, icpPts, Centre, ang1, gaps1);

    static const double A1[9][6] = {
      {12,  9, 10,  5, 16,  3},
      {32, 36, 25, 40, 29, 33},
      {57, 54, 35, 50, 61, 42},
      {78, 66, 55, 85, 74, 48},
      {102, 81, 80, 95,104, 87},
      {123, 99,100,110,106, 93},
      {138,126,125,130,119,132},
      {147,144,145,140,151,138},
      {168,171,170,175,164,177}
    };
    vector<vector<double>> ref1(9, vector<double>(6));
    for (int r=0;r<9;++r) for(int c=0;c<6;++c) ref1[r][c]=A1[r][c];

    vector<Point2d> p1f; vector<double> a1f;
    filterByReference(p1, ang1, ref1, true, p1f, a1f);
    refineByRms(p1f, a1f, ref1);

    // Choose best reference column using all plate1 angles (ang1) for robustness
    auto rms1All = computeRmsToRef(ang1, ref1);
    int bestRef1All = (int)(min_element(rms1All.begin(),rms1All.end()) - rms1All.begin());

    static const int P1idx[6][9] = {
      {5,4,6,3,7,2,1,8,9},
      {8,9,7,1,2,6,3,5,4},
      {2,1,3,9,4,8,5,7,6},
      {4,5,3,6,2,1,7,9,8},
      {7,8,6,9,1,5,2,4,3},
      {9,1,8,2,7,3,6,4,5}
    };

    // Map ALL plate1 hull points using nearest reference angle, keeping one per label by best fit
    vector<int> bestIdxPerLabel(18, -1);            // labels 1..9 used
    vector<double> bestDiffPerLabel(18, 1e12);

    for (size_t i=0;i<p1.size();++i) {
        double angle = ang1[i];
        int bestR = -1; double bestD = 1e12;
        for (int r=0;r<9;++r) {
            double d = fabs(angle - ref1[r][bestRef1All]);
            if (d < bestD) { bestD = d; bestR = r; }
        }
        int label = P1idx[bestRef1All][bestR];
        if (label>=1 && label<=9 && bestD < bestDiffPerLabel[label]) {
            bestDiffPerLabel[label] = bestD;
            bestIdxPerLabel[label] = (int)i;
        }
    }

    vector<LabeledPoint> plate1Final;
    plate1Final.reserve(9);
    for (int lbl=1; lbl<=9; ++lbl) {
        int idx = bestIdxPerLabel[lbl];
        if (idx >= 0) {
            plate1Final.push_back({p1[idx].x, p1[idx].y, lbl});
        }
    }

    // ===== Plate2 Indexing =====
    vector<double> gaps2, ang2;
    selectBestColumn(p2, icpPts, Centre, ang2, gaps2);

    static const double A2[8][6] = {
      {  1, 20,  3,  2,  8, 11},
      { 44, 26, 42, 43, 37, 34},
      { 46, 65, 47, 48, 53, 56},
      { 88, 71, 87, 88, 82, 79},
      { 91,110, 92, 93, 98,101},
      {133,116,133,132,126,124},
      {136,161,137,138,143,146},
      {178,169,178,177,171,169}
    };
    vector<vector<double>> ref2(8, vector<double>(6));
    for (int r=0;r<8;++r) for(int c=0;c<6;++c) ref2[r][c]=A2[r][c];

    vector<Point2d> p2f; vector<double> a2f;
    filterByReference(p2, ang2, ref2, false, p2f, a2f);
    refineByRms(p2f, a2f, ref2);

    auto rms2 = computeRmsToRef(a2f, ref2);
    int bestRef2 = (int)(min_element(rms2.begin(),rms2.end())-rms2.begin());

    static const int P2idx[6][8] = {
      {16,17,15,10,14,11,13,12},
      {13,12,14,11,15,10,17,16},
      {10,11,17,12,16,13,15,14},
      {12,11,13,10,14,17,15,16},
      {15,16,14,17,13,10,12,11},
      {17,16,10,15,11,14,12,13}
    };

    vector<LabeledPoint> plate2Final;
    safe_reserve(plate2Final, p2f.size());
    for (size_t i=0;i<a2f.size();++i) {
        for (int r=0;r<8;++r) {
            double refA = ref2[r][bestRef2];
            if (a2f[i] > refA-3 && a2f[i] < refA+3) {
                plate2Final.push_back({p2f[i].x, p2f[i].y, P2idx[bestRef2][r]});
                break;
            }
        }
    }

    // ===== ICP Fiducials Indexing (Following MATLAB Logic) =====
    int numIcpFid = (int)icpFid.size();
    int K = (int)plate1Final.size();
    
    // Safety check: need at least some points to proceed
    if (numIcpFid < 2 || K < 1) {
        // Return what we have so far
        vector<LabeledPoint> all;
        all.insert(all.end(), plate1Final.begin(), plate1Final.end());
        all.insert(all.end(), plate2Final.begin(), plate2Final.end());
        return all;
    }
    
    // Compute angles between ICP fiducials and plate1 points (MATLAB lines 681-703)
    // icpOrdAng(i,k) = angle between icpFid[i] and plate1Final[k]
    // MATLAB uses size(icpFid,1)-1; skip the last ICP fiducial for angle matching
    vector<vector<double>> icpOrdAng(max(0,numIcpFid-1), vector<double>(K, 0.0));
    for (int k=0; k<K; ++k) {
        Point2d plate1Pt(plate1Final[k].x, plate1Final[k].y);
        for (int i=0; i<numIcpFid-1; ++i) {
            Point2d icpPt(icpFid[i][1], icpFid[i][2]);
            
            Point2d n1 = normalizeVec(icpPt - Centre);
            Point2d n2 = normalizeVec(plate1Pt - Centre);
            icpOrdAng[i][k] = atan2d(fabs(det2d(n2,n1)), dot2d(n1,n2));
        }
    }

    // Reference angles from MATLAB (lines 705-716) - organized as columns
    static const double icpOrdAngRef[6][9] = {
      {9, 3, 16, 25, 10, 57, 35, 12, 5},    // Row 0 of each reference column
      {29, 36, 54, 33, 48, 61, 50, 40, 33}, // Row 1 of each reference column
      {42, 55, 87, 66, 81, 85, 78, 106, 80}, // Row 2 of each reference column
      {100, 74, 102, 104, 95, 99, 93, 125, 138}, // Row 3 of each reference column
      {147, 140, 130, 110, 119, 132, 126, 144, 150}, // Row 4 of each reference column
      {175, 168, 145, 138, 123, 170, 164, 177, 171}  // Row 5 of each reference column
    };

    // Find best matching (k, refCol) pair by minimum RMS (align with MATLAB loop using disComp)
    int bestRefCol = 0;
    int bestK = 0;
    double bestRms = 1e12;

    for (int k=0; k<K; ++k) {
        // Extract and sort column k
        vector<double> colAngles(max(0,numIcpFid-1));
        for (int i=0; i<numIcpFid-1; ++i) colAngles[i] = icpOrdAng[i][k];
        sort(colAngles.begin(), colAngles.end());

        // Compare against each of the 6 reference columns
        for (int refCol=0; refCol<6; ++refCol) {
            vector<double> refAngles(6);
            for (int r=0; r<6; ++r) refAngles[r] = icpOrdAngRef[r][refCol];

            auto missPoints = nearestpoint(colAngles, refAngles);
            vector<double> newRefPoints; safe_reserve(newRefPoints, missPoints.size());
            for (int idx : missPoints) newRefPoints.push_back(refAngles[idx]);

            double rms = 0.0;
            for (int i=0; i<(int)min(colAngles.size(), newRefPoints.size()); ++i) {
                double diff = newRefPoints[i] - colAngles[i];
                rms += diff * diff;
            }
            rms = sqrt(rms);

            if (rms < bestRms) { bestRms = rms; bestRefCol = refCol; bestK = k; }
        }
    }

    // ICP index mapping from MATLAB (lines 750-762) - organized as columns
    static const int icpIdxOrd[6][9] = {
      {5, 6, 4, 1, 1, 3, 1, 3, 2},  // Row 0: icpIdxOrd1[0], icpIdxOrd2[0], etc.
      {4, 5, 5, 6, 6, 4, 2, 2, 3},  // Row 1: icpIdxOrd1[1], icpIdxOrd2[1], etc.
      {6, 1, 6, 5, 5, 2, 3, 4, 1},  // Row 2: icpIdxOrd1[2], icpIdxOrd2[2], etc.
      {1, 4, 3, 4, 2, 5, 6, 1, 6},  // Row 3: icpIdxOrd1[3], icpIdxOrd2[3], etc.
      {3, 2, 2, 2, 4, 6, 5, 5, 4},  // Row 4: icpIdxOrd1[4], icpIdxOrd2[4], etc.
      {2, 3, 1, 3, 3, 1, 4, 6, 5}   // Row 5: icpIdxOrd1[5], icpIdxOrd2[5], etc.
    };

    vector<LabeledPoint> icpPlatFid;
    
    // Always perform ICP mapping with the best (k, refCol) pair
    {
        // Use the bestK column for angle matching (mirrors MATLAB sorting by icpAng then using that alignment)
        vector<double> icpAngles(max(0,numIcpFid-1));
        for (int i=0; i<numIcpFid-1; ++i) icpAngles[i] = icpOrdAng[i][bestK];

        for (int i=0; i<numIcpFid-1; ++i) {
            double angle = icpAngles[i];
            for (int r=0; r<6; ++r) {
                double refAngle = icpOrdAngRef[r][bestRefCol];
                if (angle > refAngle-2 && angle < refAngle+2) { // ±2 tolerance
                    int icpIdx = icpIdxOrd[r][bestRefCol] + 17;
                    icpPlatFid.push_back({
                        static_cast<float>(icpFid[i][1]),
                        static_cast<float>(icpFid[i][2]),
                        icpIdx
                    });
                    break;
                }
            }
        }
    }

    // Add the 6th ICP fiducial with label 23 (hack to complete set)
    if (numIcpFid == 6) {
        icpPlatFid.push_back({
            static_cast<float>(icpFid[5][1]),
            static_cast<float>(icpFid[5][2]),
            23
        });
    }

    // concatenate all
    vector<LabeledPoint> all;
    all.insert(all.end(), plate1Final.begin(), plate1Final.end());
    all.insert(all.end(), plate2Final.begin(), plate2Final.end());
    all.insert(all.end(), icpPlatFid.begin(), icpPlatFid.end());
    return all;
}

// ---------- Public Wrapper ----------
PlateFiducials plate12withICP_simple(
    const vector<Point2d>& refinedPlate1,
    const vector<Point2d>& refinedPlate2,
    const vector<vector<float>>& Z3,
    const Point2d& Centre
) {
    // convert Z3 → Vec3d
    vector<Vec3d> icpRaw;
    safe_reserve(icpRaw, Z3.size());
    for (auto &row : Z3) {
        if (row.size() >= 3) {
            icpRaw.emplace_back((double)row[0],
                                (double)row[1],
                                (double)row[2]);
        }
    }

    // run streamed ICP indexing
    auto all = icp_stream(refinedPlate1,
                          refinedPlate2,
                          icpRaw,
                          Centre);

    // split by label ranges
    PlateFiducials out;
    safe_reserve(out.final_plate1, all.size());
    safe_reserve(out.final_plate2, all.size());
    safe_reserve(out.icpPlatFid, all.size());

    for (auto &p : all) {
        if (p.label >= 1 && p.label <= 9)
            out.final_plate1.push_back(p);
        else if (p.label >= 10 && p.label <= 17)
            out.final_plate2.push_back(p);
        else
            out.icpPlatFid.push_back(p);
    }
    return out;
}

// ---------- Integration Function for Main Pipeline ----------
PlateFiducials plate12withICP_post(
    const PlateFiducials& plateIn,
    const std::vector<std::vector<float>>& Z3,
    const cv::Point2d& Centre
) {
    // Convert input PlateFiducials to Point2d vectors
    vector<Point2d> refinedPlate1, refinedPlate2;
    
    // Convert from LabeledPoint to Point2d
    safe_reserve(refinedPlate1, plateIn.final_plate1.size());
    for (const auto& pt : plateIn.final_plate1) {
        refinedPlate1.emplace_back(static_cast<double>(pt.x), static_cast<double>(pt.y));
    }
    
    safe_reserve(refinedPlate2, plateIn.final_plate2.size());
    for (const auto& pt : plateIn.final_plate2) {
        refinedPlate2.emplace_back(static_cast<double>(pt.x), static_cast<double>(pt.y));
    }
    
    // Call the main indexing function
    return plate12withICP_simple(refinedPlate1, refinedPlate2, Z3, Centre);
}