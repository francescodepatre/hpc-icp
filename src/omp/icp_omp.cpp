#include "icp_omp.h"

#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

#include "metrics.h"

using Mat4 = std::array<double, 16>;


static Mat4 identity4()
{
    Mat4 M{};

    M[0]  = 1.0;
    M[5]  = 1.0;
    M[10] = 1.0;
    M[15] = 1.0;

    return M;
}


static Mat4 to4x4(const double m[12])
{
    Mat4 M = identity4();

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            M[4 * i + j] = m[4 * i + j];
        }
    }

    return M;
}


static bool load_kitti_poses(
    const std::string& path,
    std::vector<Mat4>& poses)
{
    std::ifstream f(path);

    if (!f.is_open()) {
        std::cerr << "Unable to open poses file: "
                  << path << '\n';
        return false;
    }

    poses.clear();

    std::string line;

    while (std::getline(f, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }

        std::istringstream iss(line);

        double m[12];

        for (int i = 0; i < 12; ++i) {
            if (!(iss >> m[i])) {
                std::cerr << "Invalid line in poses file: "
                          << path << '\n';
                return false;
            }
        }

        poses.push_back(to4x4(m));
    }

    return !poses.empty();
}


static bool load_kitti_calib(
    const std::string& path,
    Mat4& Tr)
{
    std::ifstream f(path);

    if (!f.is_open()) {
        std::cerr << "Unable to open calibration file: "
                  << path << '\n';
        return false;
    }

    std::string line;

    while (std::getline(f, line)) {
        if (line.rfind("Tr:", 0) != 0) {
            continue;
        }

        std::istringstream iss(line.substr(3));

        double m[12];

        for (int i = 0; i < 12; ++i) {
            if (!(iss >> m[i])) {
                std::cerr << "Invalid Tr matrix in: "
                          << path << '\n';
                return false;
            }
        }

        Tr = to4x4(m);

        return true;
    }

    std::cerr << "Tr matrix not found in: "
              << path << '\n';

    return false;
}


static Mat4 multiply4(const Mat4& A, const Mat4& B)
{
    Mat4 C{};

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                C[4 * i + j] +=
                    A[4 * i + k] * B[4 * k + j];
            }
        }
    }

    return C;
}


static Mat4 inverse_rigid4(const Mat4& T)
{
    Mat4 inv = identity4();


    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            inv[4 * i + j] = T[4 * j + i];
        }
    }


    for (int i = 0; i < 3; ++i) {
        inv[4 * i + 3] = 0.0;

        for (int j = 0; j < 3; ++j) {
            inv[4 * i + 3] -=
                inv[4 * i + j] * T[4 * j + 3];
        }
    }

    return inv;
}


static Rigid to_rigid(const Mat4& M)
{
    Rigid T{};

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            T.R[3 * i + j] = M[4 * i + j];
        }

        T.t[i] = M[4 * i + 3];
    }

    return T;
}


namespace {

const long long MAX_CELLS = 400000000LL;
const size_t MAX_HIST_ENTRIES = 50ull * 1000 * 1000;

inline int cell_coord(float v, float minv, float inv) {
    return static_cast<int>(std::floor((v - minv) * inv));
}

struct alignas(64) PaddedSums {
    CorrSums s;
};

void merge_sums(CorrSums& a, const CorrSums& b) {
    a.n += b.n;
    for (int k = 0; k < 3; ++k) { a.sp[k] += b.sp[k]; a.sq[k] += b.sq[k]; }
    for (int k = 0; k < 9; ++k) a.spq[k] += b.spq[k];
}

}

void grid_build_omp(const PointCloud& pts, float cell, Grid& g) {
    const size_t n = pts.size();
    if (n == 0) throw std::runtime_error("grid_build_omp: empty cloud");
    if (cell <= 0.f) throw std::runtime_error("grid_build_omp: cell <= 0");
    if (n >= 2147483647ull) throw std::runtime_error("grid_build_omp: too many points for int indices");

    float minx = pts.x[0], miny = pts.y[0], minz = pts.z[0];
    float maxx = minx, maxy = miny, maxz = minz;
#pragma omp parallel for schedule(static) reduction(min : minx, miny, minz) reduction(max : maxx, maxy, maxz)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        minx = std::min(minx, pts.x[i]); maxx = std::max(maxx, pts.x[i]);
        miny = std::min(miny, pts.y[i]); maxy = std::max(maxy, pts.y[i]);
        minz = std::min(minz, pts.z[i]); maxz = std::max(maxz, pts.z[i]);
    }
    g.minx = minx; g.miny = miny; g.minz = minz;

    g.cell = cell;
    g.inv = 1.0f / cell;
    g.nx = cell_coord(maxx, g.minx, g.inv) + 1;
    g.ny = cell_coord(maxy, g.miny, g.inv) + 1;
    g.nz = cell_coord(maxz, g.minz, g.inv) + 1;

    const long long ncells = static_cast<long long>(g.nx) * g.ny * g.nz;
    if (ncells > MAX_CELLS)
        throw std::runtime_error("grid_build_omp: " + std::to_string(ncells) +
                                 " cells: too many. Increase the cell size or downsample.");

    const int maxT = omp_get_max_threads();
    if (static_cast<size_t>(maxT) * static_cast<size_t>(ncells) > MAX_HIST_ENTRIES) {
        grid_build(pts, cell, g);
        return;
    }

    std::vector<int> cid(n);
    std::vector<int> cnt(static_cast<size_t>(maxT) * static_cast<size_t>(ncells), 0);
    g.cellStart.assign(static_cast<size_t>(ncells) + 1, 0);
    g.x.resize(n); g.y.resize(n); g.z.resize(n); g.orig.resize(n);

#pragma omp parallel num_threads(maxT)
    {
        const int t = omp_get_thread_num();
        const int T = omp_get_num_threads();
        const size_t lo = n * static_cast<size_t>(t) / static_cast<size_t>(T);
        const size_t hi = n * static_cast<size_t>(t + 1) / static_cast<size_t>(T);
        int* mine = cnt.data() + static_cast<size_t>(t) * static_cast<size_t>(ncells);

        for (size_t i = lo; i < hi; ++i) {
            int ix = std::min(cell_coord(pts.x[i], g.minx, g.inv), g.nx - 1);
            int iy = std::min(cell_coord(pts.y[i], g.miny, g.inv), g.ny - 1);
            int iz = std::min(cell_coord(pts.z[i], g.minz, g.inv), g.nz - 1);
            const int c = ix + g.nx * (iy + g.ny * iz);
            cid[i] = c;
            mine[c]++;
        }
#pragma omp barrier

#pragma omp for schedule(static)
        for (long long c = 0; c < ncells; ++c) {
            int run = 0;
            for (int u = 0; u < T; ++u) {
                int& e = cnt[static_cast<size_t>(u) * static_cast<size_t>(ncells) + static_cast<size_t>(c)];
                const int v = e;
                e = run;
                run += v;
            }
            g.cellStart[static_cast<size_t>(c) + 1] = run;
        }
#pragma omp single
        {
            for (long long c = 0; c < ncells; ++c) g.cellStart[c + 1] += g.cellStart[c];
        }
        for (size_t i = lo; i < hi; ++i) {
            const int c = cid[i];
            const int pos = g.cellStart[c] + mine[c]++;
            g.x[pos] = pts.x[i];
            g.y[pos] = pts.y[i];
            g.z[pos] = pts.z[i];
            g.orig[pos] = static_cast<int>(i);
        }
    }
}

IcpResult icp_omp(const PointCloud& src, const PointCloud& tgt, const IcpParams& p, const Rigid& init) {
    IcpResult res;
    res.T = init;
    Timer total;
    Timer t;

    Grid grid;
    t.tic();
    grid_build_omp(tgt, p.dmax, grid);
    res.tm.grid = t.toc();

    const long long n = static_cast<long long>(src.size());
    PointCloud moved;
    moved.resize(src.size());
    std::vector<int> match(src.size());
    const int maxT = omp_get_max_threads();
    std::vector<PaddedSums> parts(static_cast<size_t>(maxT));

    for (int it = 0; it < p.iters; ++it) {

        t.tic();
        {
            const Rigid T = res.T;
#pragma omp parallel for schedule(static)
            for (long long i = 0; i < n; ++i) {
                double o[3];
                rigid_apply(T, src.x[i], src.y[i], src.z[i], o);
                moved.x[i] = static_cast<float>(o[0]);
                moved.y[i] = static_cast<float>(o[1]);
                moved.z[i] = static_cast<float>(o[2]);
            }
        }
        res.tm.apply += t.toc();

        t.tic();
        double sum_d2 = 0;
        size_t valid = 0;
#pragma omp parallel for schedule(runtime) reduction(+ : sum_d2, valid)
        for (long long i = 0; i < n; ++i) {
            float d2 = 0;
            match[i] = grid_nearest(grid, moved.x[i], moved.y[i], moved.z[i], p.dmax, &d2);
            if (match[i] >= 0) { sum_d2 += d2; ++valid; }
        }
        res.tm.nn += t.toc();

        t.tic();
        for (auto& ps : parts) ps.s = CorrSums();
#pragma omp parallel num_threads(maxT)
        {
            CorrSums local;
#pragma omp for schedule(static) nowait
            for (long long i = 0; i < n; ++i) {
                const int j = match[i];
                if (j < 0) continue;
                local.add(moved.x[i], moved.y[i], moved.z[i], grid.x[j], grid.y[j], grid.z[j]);
            }
            parts[static_cast<size_t>(omp_get_thread_num())].s = local;
        }
        CorrSums s;
        for (const auto& ps : parts) merge_sums(s, ps.s);
        res.tm.reduce += t.toc();

        t.tic();
        res.rmse.push_back(valid ? std::sqrt(sum_d2 / static_cast<double>(valid)) : 0.0);
        res.matches.push_back(valid);
        Rigid delta;
        if (!kabsch_from_sums(s, delta)) {
            res.tm.solve += t.toc();
            break;
        }
        res.T = rigid_compose(delta, res.T);
        res.tm.solve += t.toc();
    }

    res.tm.total = total.toc();
    return res;
}