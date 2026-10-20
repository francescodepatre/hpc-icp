#include <omp.h>

#include <cmath>
#include <cstdio>
#include <random>

#include "grid.h"
#include "icp.h"
#include "icp_omp.h"
#include "kabsch.h"
#include "metrics.h"

static int failures = 0;
#define CHECK(cond, ...)                              \
    do {                                              \
        if (!(cond)) {                                \
            ++failures;                               \
            std::printf("  FALLITO: %s  -> ", #cond); \
            std::printf(__VA_ARGS__);                 \
            std::printf("\n");                        \
        }                                             \
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

static bool same_grid(const Grid& a, const Grid& b) {
    return a.nx == b.nx && a.ny == b.ny && a.nz == b.nz && a.minx == b.minx && a.miny == b.miny &&
           a.minz == b.minz && a.cellStart == b.cellStart && a.x == b.x && a.y == b.y && a.z == b.z &&
           a.orig == b.orig;
}

int main() {
    const int thread_counts[] = {1, 2, 3, 4, 7, 16};

    std::printf("[griglia OpenMP == griglia sequenziale]\n");
    PointCloud cloud = make_surface(60000, 1);
    {   // aggiungo un cluster molto denso: celle con carichi molto diversi
        std::mt19937 rng(9);
        std::normal_distribution<float> g(0.2f, 0.02f);
        for (int i = 0; i < 20000; ++i) cloud.push_back(g(rng), g(rng), g(rng));
    }
    Grid gs;
    grid_build(cloud, 0.1f, gs);
    for (int T : thread_counts) {
        omp_set_num_threads(T);
        Grid go;
        grid_build_omp(cloud, 0.1f, go);
        const bool ok = same_grid(gs, go);
        std::printf("  %2d thread: %s\n", T, ok ? "identica" : "DIVERSA");
        CHECK(ok, "griglia diversa con %d thread", T);
    }

    std::printf("[ICP OpenMP vs sequenziale]\n");
    PointCloud tgt = make_surface(10000, 11), base = make_surface(10000, 12), src;
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
    prm.iters = 30;
    IcpResult rs = icp_seq(src, tgt, prm);
    std::printf("  seq        : %.3f s\n", rs.tm.total);

    for (int T : thread_counts) {
        omp_set_num_threads(T);
        IcpResult ro = icp_omp(src, tgt, prm);
        double rot, tr;
        pose_error(ro.T, rs.T, rot, tr);
        const double drmse = std::fabs(ro.rmse.back() - rs.rmse.back());
        std::printf("  %2d thread  : %.3f s | diff da seq: rot %.2e gradi, trasl %.2e, rmse %.2e, coppie %zu vs %zu\n",
                    T, ro.tm.total, rot, tr, drmse, ro.matches.back(), rs.matches.back());
        CHECK(rot < 1e-6 && tr < 1e-6, "T diversa dalla sequenziale con %d thread", T);
        CHECK(drmse < 1e-6, "RMSE diverso con %d thread", T);
        CHECK(ro.matches.back() == rs.matches.back(), "numero di coppie diverso con %d thread", T);
    }

    std::printf("\n%s (%d controlli falliti)\n", failures ? "TEST FALLITI" : "TUTTI I TEST OK", failures);
    return failures ? 1 : 0;
}
