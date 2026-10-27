#pragma once

#include "icp.h"
#include "voxel.h"

#include <stdexcept>

inline Rigid compose_transforms(const Rigid& A, const Rigid& B) {
    Rigid C{};

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            C.R[3 * i + j] = 0.0;

            for (int k = 0; k < 3; ++k) {
                C.R[3 * i + j] +=
                    A.R[3 * i + k] * B.R[3 * k + j];
            }
        }

        C.t[i] = A.t[i];

        for (int k = 0; k < 3; ++k) {
            C.t[i] += A.R[3 * i + k] * B.t[k];
        }
    }

    return C;
}

inline void append_transformed(
    PointCloud& map,
    const PointCloud& cloud,
    const Rigid& T
) {
    const size_t old_size = map.size();

    map.resize(old_size + cloud.size());

    for (size_t i = 0; i < cloud.size(); ++i) {
        double p[3];

        rigid_apply(
            T,
            cloud.x[i],
            cloud.y[i],
            cloud.z[i],
            p
        );

        map.x[old_size + i] = static_cast<float>(p[0]);
        map.y[old_size + i] = static_cast<float>(p[1]);
        map.z[old_size + i] = static_cast<float>(p[2]);
    }
}