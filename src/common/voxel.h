#pragma once

#ifndef VOXEL_H
#define VOXEL_H
// Pezzo 2: voxel downsampling.
// Divide lo spazio in voxel cubici di lato `voxel` e sostituisce i punti di ciascun
// voxel con il loro centroide. L'output e' ordinato per voxel (deterministico).
#include "PointCloud.h"

PointCloud voxel_downsample(const PointCloud& in, float voxel);

#endif