#pragma once
#include <vector>

std::vector<int> DBSCAN(const std::vector<double>& X, double epsilon, int MinPts);
