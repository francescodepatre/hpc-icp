#pragma once

#ifndef ICP_MPI_H
#define ICP_MPI_H

#include "grid.h"
#include "icp.h"
#include "kabsch.h"

enum class Partition {
    Block,  
    Cyclic   
};

enum class Collective {
    ReduceBcast,  
    Allreduce     
};

struct MpiOptions {
    Partition partition = Partition::Block;
    Collective collective = Collective::ReduceBcast;
};

struct MpiStats {
    double compute = 0;        
    double comm = 0;             
    size_t local_points = 0;     
};

IcpResult icp_mpi(const PointCloud& src, const PointCloud& tgt, const IcpParams& p,
                  const Rigid& init = rigid_identity(), const MpiOptions& opt = MpiOptions(),
                  MpiStats* stats = nullptr);

void bcast_cloud(PointCloud& pc, int root = 0);

#ifndef