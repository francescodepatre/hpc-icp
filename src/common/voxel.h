#pragma once

#ifndef VOXEL_H
#define VOXEL_H

#include "PointCloud.h"

PointCloud voxel_downsample(const PointCloud& in, float voxel);

#endif