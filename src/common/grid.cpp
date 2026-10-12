#include "grid.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {
const long long MAX_CELLS = 400000000LL;  // safety limit on memory (~1.6 GB di int)

inline int cell_coord(float v, float minv, float inv) {
    return static_cast<int>(std::floor((v - minv) * inv));
}
}  // namespace

void grid_build(const PointCloud& pts, float cell, Grid& g) {
    const size_t n = pts.size();
    if (n == 0) throw std::runtime_error("grid_build: nuvola vuota");
    if (cell <= 0.f) throw std::runtime_error("grid_build: cell <= 0");
    if (n >= 2147483647ull) throw std::runtime_error("grid_build: troppi punti per indici int");

    float maxx = pts.x[0], maxy = pts.y[0], maxz = pts.z[0];
    g.minx = pts.x[0]; g.miny = pts.y[0]; g.minz = pts.z[0];
    for (size_t i = 1; i < n; ++i) {
        g.minx = std::min(g.minx, pts.x[i]); maxx = std::max(maxx, pts.x[i]);
        g.miny = std::min(g.miny, pts.y[i]); maxy = std::max(maxy, pts.y[i]);
        g.minz = std::min(g.minz, pts.z[i]); maxz = std::max(maxz, pts.z[i]);
    }

    g.cell = cell;
    g.inv = 1.0f / cell;
    g.nx = cell_coord(maxx, g.minx, g.inv) + 1;
    g.ny = cell_coord(maxy, g.miny, g.inv) + 1;
    g.nz = cell_coord(maxz, g.minz, g.inv) + 1;

    const long long ncells = static_cast<long long>(g.nx) * g.ny * g.nz;
    if (ncells > MAX_CELLS)
        throw std::runtime_error("grid_build: " + std::to_string(ncells) +
                                 " celle: troppe. Aumenta il lato della cella o fai downsampling.");

    // Passo 1: indice di cella di ogni punto e istogramma
    std::vector<int> cid(n);
    g.cellStart.assign(static_cast<size_t>(ncells) + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        int ix = std::min(cell_coord(pts.x[i], g.minx, g.inv), g.nx - 1);
        int iy = std::min(cell_coord(pts.y[i], g.miny, g.inv), g.ny - 1);
        int iz = std::min(cell_coord(pts.z[i], g.minz, g.inv), g.nz - 1);
        cid[i] = ix + g.nx * (iy + g.ny * iz);
        g.cellStart[cid[i] + 1]++;
    }

    // Passo 2: prefix sum -> cellStart[c] e' l'inizio della cella c
    for (long long c = 0; c < ncells; ++c) g.cellStart[c + 1] += g.cellStart[c];

    // Passo 3: scatter dei punti nella posizione ordinata
    g.x.resize(n); g.y.resize(n); g.z.resize(n); g.orig.resize(n);
    std::vector<int> cursor(g.cellStart.begin(), g.cellStart.end() - 1);
    for (size_t i = 0; i < n; ++i) {
        const int pos = cursor[cid[i]]++;
        g.x[pos] = pts.x[i];
        g.y[pos] = pts.y[i];
        g.z[pos] = pts.z[i];
        g.orig[pos] = static_cast<int>(i);
    }
}

int grid_nearest(const Grid& g, float px, float py, float pz, float dmax, float* d2out) {
    const float fx = std::floor((px - g.minx) * g.inv);
    const float fy = std::floor((py - g.miny) * g.inv);
    const float fz = std::floor((pz - g.minz) * g.inv);

    // Punto troppo lontano dal bounding box: nessun vicino possibile (e niente overflow nel cast)
    if (fx < -1.f || fx > static_cast<float>(g.nx) ||
        fy < -1.f || fy > static_cast<float>(g.ny) ||
        fz < -1.f || fz > static_cast<float>(g.nz))
        return -1;

    const int ix = static_cast<int>(fx), iy = static_cast<int>(fy), iz = static_cast<int>(fz);
    float best = dmax * dmax;
    int best_j = -1;

    const int z0 = std::max(iz - 1, 0), z1 = std::min(iz + 1, g.nz - 1);
    const int y0 = std::max(iy - 1, 0), y1 = std::min(iy + 1, g.ny - 1);
    const int x0 = std::max(ix - 1, 0), x1 = std::min(ix + 1, g.nx - 1);

    for (int cz = z0; cz <= z1; ++cz)
        for (int cy = y0; cy <= y1; ++cy)
            for (int cx = x0; cx <= x1; ++cx) {
                const int c = cx + g.nx * (cy + g.ny * cz);
                for (int j = g.cellStart[c]; j < g.cellStart[c + 1]; ++j) {
                    const float dx = g.x[j] - px;
                    const float dy = g.y[j] - py;
                    const float dz = g.z[j] - pz;
                    const float d2 = dx * dx + dy * dy + dz * dz;
                    if (d2 < best) { best = d2; best_j = j; }
                }
            }

    if (best_j >= 0 && d2out) *d2out = best;
    return best_j;
}

int brute_nearest(const PointCloud& pts, float px, float py, float pz, float dmax, float* d2out) {
    float best = dmax * dmax;
    int best_i = -1;
    for (size_t i = 0; i < pts.size(); ++i) {
        const float dx = pts.x[i] - px;
        const float dy = pts.y[i] - py;
        const float dz = pts.z[i] - pz;
        const float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < best) { best = d2; best_i = static_cast<int>(i); }
    }
    if (best_i >= 0 && d2out) *d2out = best;
    return best_i;
}
