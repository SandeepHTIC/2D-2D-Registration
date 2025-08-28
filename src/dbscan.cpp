#include "DBSCAN.h"
#include <cmath>
#include <vector>

std::vector<int> DBSCAN(const std::vector<double>& X, double epsilon, int MinPts) {
    int n = X.size();
    std::vector<int> IDX(n, 0);               // Cluster labels: 0 = unassigned
    std::vector<std::vector<bool>> D(n, std::vector<bool>(n, false));  // Distance matrix
    std::vector<bool> visited(n, false);
    int C = 0;

    // === Build binary distance matrix D(i,j) = abs(X[i] - X[j]) <= epsilon ===
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            D[i][j] = std::fabs(X[i] - X[j]) <= epsilon;

    // === Main DBSCAN Loop ===
    for (int i = 0; i < n; ++i) {
        if (visited[i]) continue;
        visited[i] = true;

        // Find neighbors of i
        std::vector<int> Neighbors;
        for (int j = 0; j < n; ++j)
            if (D[i][j]) Neighbors.push_back(j);

        if (Neighbors.size() < MinPts)
            continue;  // Don't mark as noise, just skip
        else {
            C++;
            // Expand cluster
            IDX[i] = C;
            int k = 0;
            while (k < Neighbors.size()) {
                int j = Neighbors[k];
                if (!visited[j]) {
                    visited[j] = true;

                    // RegionQuery(j)
                    std::vector<int> Neighbors2;
                    for (int m = 0; m < n; ++m)
                        if (D[j][m]) Neighbors2.push_back(m);

                    if (Neighbors2.size() >= MinPts) {
                        Neighbors.insert(Neighbors.end(), Neighbors2.begin(), Neighbors2.end());
                    }
                }
                if (IDX[j] == 0)
                    IDX[j] = C;
                ++k;
            }
        }
    }

    return IDX;
}
