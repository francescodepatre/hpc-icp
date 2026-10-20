#pragma once

#ifndef ICP_H
#define ICP_H
#include <vector>
#include "grid.h"
#include "kabsch.h"
#include "PointCloud.h"

struct IcpParams {
    float dmax = 0.5f;  
    int iters = 30;     
};

struct IcpTimings {
    double grid = 0;    
    double apply = 0, nn = 0, reduce = 0, solve = 0;  
    double total = 0;
};

struct IcpResult {
    Rigid T;                        
    IcpTimings tm;
    std::vector<double> rmse;       
    std::vector<size_t> matches;    
};

IcpResult icp_seq(const PointCloud& src, const PointCloud& tgt, const IcpParams& p,
                  const Rigid& init = rigid_identity());

#endif