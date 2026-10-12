#pragma once

#ifndef GRID_H
#define GRID_H

#include <vector>
#include "PointCloud.h"

struct Grid {
    float minx = 0, miny = 0, minz = 0;
    float cell = 1, inv = 1;
    int nx = 0, ny = 0, nz = 0;
    std::vector<int> cellStart;
    std::vector<float> x, y, z;
    std::vector<int> orig;
};

void grid_build(const PointCloud& pts, float cell, Grid& g);

int grid_nearest(const Grid& g, float px, float py, float pz, float dmax, float* d2out);

int brute_nearest(const PointCloud& pts, float px, float py, float pz, float dmax, float* d2out);

#endif