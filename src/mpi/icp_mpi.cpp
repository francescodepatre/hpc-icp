#include "icp_mpi.h"

#ifndef _OPENMP
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#endif

#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#ifndef MPICH_SKIP_MPICXX
#define MPICH_SKIP_MPICXX 1
#endif
#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <stdexcept>
#include <vector>

#include "metrics.h"

namespace {

constexpr int NBUF = 18;
constexpr int NMSG = 15;

void pack(const CorrSums& s, double sum_d2, double valid, double* b) {
    b[0] = s.n;
    for (int k = 0; k < 3; ++k) { b[1 + k] = s.sp[k]; b[4 + k] = s.sq[k]; }
    for (int k = 0; k < 9; ++k) b[7 + k] = s.spq[k];
    b[16] = sum_d2;
    b[17] = valid;
}

void unpack(const double* b, CorrSums& s, double& sum_d2, double& valid) {
    s.n = b[0];
    for (int k = 0; k < 3; ++k) { s.sp[k] = b[1 + k]; s.sq[k] = b[4 + k]; }
    for (int k = 0; k < 9; ++k) s.spq[k] = b[7 + k];
    sum_d2 = b[16];
    valid = b[17];
}

void solve_step(const double* g, const Rigid& cur, double* msg) {
    CorrSums s;
    double sum_d2 = 0, valid = 0;
    unpack(g, s, sum_d2, valid);

    msg[13] = valid > 0 ? std::sqrt(sum_d2 / valid) : 0.0;
    msg[14] = valid;

    Rigid next = cur, delta;
    const bool ok = kabsch_from_sums(s, delta);
    if (ok) next = rigid_compose(delta, cur);

    msg[0] = ok ? 0.0 : 1.0;
    for (int k = 0; k < 9; ++k) msg[1 + k] = next.R[k];
    for (int k = 0; k < 3; ++k) msg[10 + k] = next.t[k];
}

#ifdef _OPENMP
struct alignas(64) PaddedSums {
    CorrSums s;
};

void merge_sums(CorrSums& a, const CorrSums& b) {
    a.n += b.n;
    for (int k = 0; k < 3; ++k) { a.sp[k] += b.sp[k]; a.sq[k] += b.sq[k]; }
    for (int k = 0; k < 9; ++k) a.spq[k] += b.spq[k];
}
#endif

CorrSums local_sums(const PointCloud& moved, const std::vector<int>& match, const Grid& grid,
                    long long nloc) {
#ifdef _OPENMP
    const int maxT = omp_get_max_threads();
    std::vector<PaddedSums> parts(static_cast<size_t>(maxT));
#pragma omp parallel num_threads(maxT)
    {
        CorrSums local;
#pragma omp for schedule(static) nowait
        for (long long k = 0; k < nloc; ++k) {
            const int j = match[k];
            if (j < 0) continue;
            local.add(moved.x[k], moved.y[k], moved.z[k], grid.x[j], grid.y[j], grid.z[j]);
        }
        parts[static_cast<size_t>(omp_get_thread_num())].s = local;
    }
    CorrSums s;
    for (const auto& ps : parts) merge_sums(s, ps.s);
    return s;
#else
    CorrSums s;
    for (long long k = 0; k < nloc; ++k) {
        const int j = match[k];
        if (j < 0) continue;
        s.add(moved.x[k], moved.y[k], moved.z[k], grid.x[j], grid.y[j], grid.z[j]);
    }
    return s;
#endif
}

}

void bcast_cloud(PointCloud& pc, int root) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    unsigned long long n = pc.size();
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG_LONG, root, MPI_COMM_WORLD);
    if (n >= 2147483647ull) throw std::runtime_error("bcast_cloud: too many points for an int count");

    if (rank != root) pc.resize(static_cast<size_t>(n));
    const int cnt = static_cast<int>(n);
    MPI_Bcast(pc.x.data(), cnt, MPI_FLOAT, root, MPI_COMM_WORLD);
    MPI_Bcast(pc.y.data(), cnt, MPI_FLOAT, root, MPI_COMM_WORLD);
    MPI_Bcast(pc.z.data(), cnt, MPI_FLOAT, root, MPI_COMM_WORLD);
}

IcpResult icp_mpi(const PointCloud& src, const PointCloud& tgt, const IcpParams& p, const Rigid& init,
                  const MpiOptions& opt, MpiStats* stats) {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (src.empty()) throw std::runtime_error("icp_mpi: empty source cloud");

    MPI_Barrier(MPI_COMM_WORLD);

    IcpResult res;
    res.T = init;
    Timer total, t;
    double t_comm = 0, t_compute = 0;

    Grid grid;
    t.tic();
    grid_build(tgt, p.dmax, grid);
    res.tm.grid = t.toc();

    const long long n = static_cast<long long>(src.size());
    long long start, stride, nloc;
    if (opt.partition == Partition::Block) {
        const long long lo = n * rank / size, hi = n * (rank + 1) / size;
        start = lo;
        stride = 1;
        nloc = hi - lo;
    } else {
        start = rank;
        stride = size;
        nloc = (n > rank) ? (n - rank + size - 1) / size : 0;
    }

    PointCloud moved;
    moved.resize(static_cast<size_t>(nloc));
    std::vector<int> match(static_cast<size_t>(nloc), -1);

    for (int it = 0; it < p.iters; ++it) {
        t.tic();
        {
            const Rigid T = res.T;
#pragma omp parallel for schedule(static)
            for (long long k = 0; k < nloc; ++k) {
                const long long i = start + k * stride;
                double o[3];
                rigid_apply(T, src.x[i], src.y[i], src.z[i], o);
                moved.x[k] = static_cast<float>(o[0]);
                moved.y[k] = static_cast<float>(o[1]);
                moved.z[k] = static_cast<float>(o[2]);
            }
        }
        double dt = t.toc();
        res.tm.apply += dt;
        t_compute += dt;

        t.tic();
        double sum_d2 = 0;
        unsigned long long valid = 0;
#pragma omp parallel for schedule(runtime) reduction(+ : sum_d2, valid)
        for (long long k = 0; k < nloc; ++k) {
            float d2 = 0;
            match[k] = grid_nearest(grid, moved.x[k], moved.y[k], moved.z[k], p.dmax, &d2);
            if (match[k] >= 0) { sum_d2 += d2; ++valid; }
        }
        dt = t.toc();
        res.tm.nn += dt;
        t_compute += dt;

        t.tic();
        const CorrSums s_local = local_sums(moved, match, grid, nloc);
        double lbuf[NBUF];
        pack(s_local, sum_d2, static_cast<double>(valid), lbuf);
        dt = t.toc();
        res.tm.reduce += dt;
        t_compute += dt;

        double gbuf[NBUF] = {};
        double msg[NMSG] = {};
        if (opt.collective == Collective::Allreduce) {
            t.tic();
            MPI_Allreduce(lbuf, gbuf, NBUF, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            t_comm += t.toc();
            t.tic();
            solve_step(gbuf, res.T, msg);
            res.tm.solve += t.toc();
        } else {
            t.tic();
            MPI_Reduce(lbuf, gbuf, NBUF, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            t_comm += t.toc();
            if (rank == 0) {
                t.tic();
                solve_step(gbuf, res.T, msg);
                res.tm.solve += t.toc();
            }

            t.tic();
            MPI_Bcast(msg, NMSG, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            t_comm += t.toc();
        }

        res.rmse.push_back(msg[13]);
        res.matches.push_back(static_cast<size_t>(msg[14]));
        if (msg[0] != 0.0) break;
        for (int k = 0; k < 9; ++k) res.T.R[k] = msg[1 + k];
        for (int k = 0; k < 3; ++k) res.T.t[k] = msg[10 + k];
    }

    res.tm.total = total.toc();
    if (stats) {
        stats->compute = t_compute;
        stats->comm = t_comm;
        stats->local_points = static_cast<size_t>(nloc);
    }
    return res;
}