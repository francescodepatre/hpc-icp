#pragma once

#ifndef ICP_OMP_H
#define ICP_OMP_H

#include "grid.h"
#include "icp.h"
#include "kabsch.h"

void grid_build_omp(
    const PointCloud& pts,
    float cell,
    Grid& g
);

IcpResult icp_omp(
    const PointCloud& src,
    const PointCloud& tgt,
    const IcpParams& p,
    const Rigid& init = rigid_identity()
);

#endif