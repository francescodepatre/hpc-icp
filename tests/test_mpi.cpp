#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#ifndef MPICH_SKIP_MPICXX
#define MPICH_SKIP_MPICXX 1
#endif
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <random>

#include "icp.h"
#include "icp_mpi.h"
#include "kabsch.h"
#include "metrics.h"

static int failures = 0;
static int g_rank = 0;
#define CHECK(cond, ...)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            ++failures;                                               \
            if (g_rank == 0) {                                        \
                std::printf("  FALLITO: %s  -> ", #cond);             \
                std::printf(__VA_ARGS__);                             \
                std::printf("\n");                                    \
            }                                                         \
        }                                                             \
    } while (0)

static Rigid make_rigid(double ax, double ay, double az, double deg, double tx, double ty, double tz) {
    const double n = std::sqrt(ax * ax + ay * ay + az * az);
    ax /= n; ay /= n; az /= n;
    const double a = deg * 3.14159265358979323846 / 180.0, c = std::cos(a), s = std::sin(a), C = 1 - c;
    Rigid T;
    T.R[0] = c + ax * ax * C;      T.R[1] = ax * ay * C - az * s; T.R[2] = ax * az * C + ay * s;
    T.R[3] = ay * ax * C + az * s; T.R[4] = c + ay * ay * C;      T.R[5] = ay * az * C - ax * s;
    T.R[6] = az * ax * C - ay * s; T.R[7] = az * ay * C + ax * s; T.R[8] = c + az * az * C;
    T.t[0] = tx; T.t[1] = ty; T.t[2] = tz;
    return T;
}

static PointCloud make_surface(size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0, 1);
    PointCloud pc;
    for (size_t i = 0; i < n; ++i) {
        const double th = std::acos(1 - 2 * u(rng)), ph = 2 * 3.14159265358979323846 * u(rng);
        const double r = 1 + 0.25 * std::sin(3 * th) * std::cos(2 * ph) + 0.1 * std::cos(5 * ph);
        pc.push_back(static_cast<float>(r * std::sin(th) * std::cos(ph)),
                     static_cast<float>(0.6 * r * std::sin(th) * std::sin(ph)),
                     static_cast<float>(0.3 * r * std::cos(th)));
    }
    return pc;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (g_rank == 0) std::printf("test_mpi with %d rank\n", size);

    PointCloud tgt = make_surface(8000, 11), base = make_surface(8000, 12), src;
    Rigid G = make_rigid(1, 2, 3, 6.0, 0.05, -0.03, 0.02), Gi = rigid_inverse(G);
    std::mt19937 rng(21);
    std::normal_distribution<double> noise(0, 0.002);
    for (size_t i = 0; i < base.size(); ++i) {
        double o[3];
        rigid_apply(Gi, base.x[i], base.y[i], base.z[i], o);
        src.push_back(static_cast<float>(o[0] + noise(rng)), static_cast<float>(o[1] + noise(rng)),
                      static_cast<float>(o[2] + noise(rng)));
    }
    IcpParams prm;
    prm.dmax = 0.15f;
    prm.iters = 25;
    const IcpResult ref = icp_seq(src, tgt, prm);

    if (g_rank == 0) std::printf("[icp_mpi == icp_seq]\n");
    const Partition parts[] = {Partition::Block, Partition::Cyclic};
    const Collective colls[] = {Collective::ReduceBcast, Collective::Allreduce};
    for (Partition pa : parts)
        for (Collective co : colls) {
            MpiOptions opt;
            opt.partition = pa;
            opt.collective = co;
            MpiStats st;
            IcpResult r = icp_mpi(src, tgt, prm, rigid_identity(), opt, &st);

            double rot, tr;
            pose_error(r.T, ref.T, rot, tr);
            const double drmse = std::fabs(r.rmse.back() - ref.rmse.back());

            unsigned long long loc = st.local_points, tot = 0;
            MPI_Allreduce(&loc, &tot, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

            if (g_rank == 0)
                std::printf("  %-6s %-11s: diff from seq rot %.2e deg, trasl %.2e, rmse %.2e, coppie %zu vs %zu, " "points %llu/%zu\n", pa == Partition::Block ? "block" : "cyclic", co == Collective::ReduceBcast ? "reduce+bcast" : "allreduce", rot, tr, drmse, r.matches.back(), ref.matches.back(), tot, src.size());
            CHECK(rot < 1e-6 && tr < 1e-6, "T different from sequential");
            CHECK(drmse < 1e-6, "RMSE diverso");
            CHECK(r.matches.back() == ref.matches.back(), "numero di coppie diverso");
            CHECK(tot == src.size(), "i punti assegnati non coprono la sorgente");
            CHECK(r.rmse.size() == ref.rmse.size(), "numero di iterazioni diverso");
        }

    {
        MpiOptions opt;
        IcpResult r = icp_mpi(src, tgt, prm, rigid_identity(), opt);
        double buf[12], root[12];
        for (int k = 0; k < 9; ++k) buf[k] = root[k] = r.T.R[k];
        for (int k = 0; k < 3; ++k) buf[9 + k] = root[9 + k] = r.T.t[k];
        MPI_Bcast(root, 12, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        bool same = true;
        for (int k = 0; k < 12; ++k) same = same && (buf[k] == root[k]);
        int s = same ? 0 : 1, any = 0;
        MPI_Allreduce(&s, &any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (g_rank == 0) std::printf("[T same on all ranks (reduce+bcast)]: %s\n", any ? "NO" : "yes");
        CHECK(!any, "T different in ranks");
    }

    {
        PointCloud tiny;
        tiny.push_back(0.f, 0.f, 0.f);
        tiny.push_back(0.1f, 0.f, 0.f);
        IcpParams p2 = prm;
        p2.iters = 3;
        MpiOptions opt;
        opt.partition = Partition::Cyclic;
        IcpResult r = icp_mpi(tiny, tgt, p2, rigid_identity(), opt);
        if (g_rank == 0) std::printf("[2 source points on %d rank]: ok, #iterations %zu\n", size, r.rmse.size());
        CHECK(r.rmse.size() == 1, "less than 3 pairs -> stop at first iter");
    }

    int bad = failures > 0 ? 1 : 0, any_bad = 0;
    MPI_Allreduce(&bad, &any_bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (g_rank == 0) std::printf("\n%s\n", any_bad ? "FAILES TESTS" : "ALL TEST OK");
    MPI_Finalize();
    return any_bad ? 1 : 0;
}
