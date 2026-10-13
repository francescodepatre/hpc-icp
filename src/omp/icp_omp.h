#pragma once
// Versione OpenMP: stessa interfaccia della versione sequenziale (icp.h / grid.h),
// cosi' i risultati sono confrontabili 1:1.
//
//  - grid_build_omp : costruzione della griglia in parallelo. Produce LO STESSO layout
//                     della versione sequenziale (stesso ordine dei punti in ogni cella).
//  - icp_omp        : ICP con le quattro fasi parallelizzate (apply, nn, reduce) e
//                     solve sequenziale (SVD/quaternione 3x3: costo trascurabile).
//
// grid_nearest() della versione sequenziale e' una funzione const senza stato condiviso,
// quindi si puo' chiamare cosi' com'e' da piu' thread.
#include "grid.h"
#include "icp.h"
#include "kabsch.h"

void grid_build_omp(const PointCloud& pts, float cell, Grid& g);

IcpResult icp_omp(const PointCloud& src, const PointCloud& tgt, const IcpParams& p,
                  const Rigid& init = rigid_identity());
