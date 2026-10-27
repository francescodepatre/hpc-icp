#pragma once

#ifndef ICP_CUDA_H
#define ICP_CUDA_H

#include <string>

#include "grid.h"
#include "icp.h"
#include "kabsch.h"

struct CudaOptions {
    int device = 0;
    int block_nn = 128;     
    int block_other = 256;  
};

struct CudaInfo {
    std::string device_name;
    int sm_count = 0;
    double t_init = 0;  
    double t_h2d = 0;   
    double t_d2h = 0;   
};

IcpResult icp_cuda(const PointCloud& src, const PointCloud& tgt, const IcpParams& p,
                   const Rigid& init = rigid_identity(), const CudaOptions& opt = CudaOptions(),
                   CudaInfo* info = nullptr);

#endif