// Test della baseline: kabsch, griglia vs brute force, voxel, I/O, ICP end-to-end.
// Esce con codice != 0 se un controllo fallisce.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>

#include "grid.h"
#include "icp.h"
#include "io.h"
#include "kabsch.h"
#include "metrics.h"
#include "voxel.h"

static int failures = 0;
#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            ++failures;                                   \
            std::printf("  FALLITO: %s  -> ", #cond);     \
            std::printf(__VA_ARGS__);                     \
            std::printf("\n");                            \
        }                                                 \
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
        pc.push_back(static_cast<float>(1.0 * r * std::sin(th) * std::cos(ph)),
                     static_cast<float>(0.6 * r * std::sin(th) * std::sin(ph)),
                     static_cast<float>(0.3 * r * std::cos(th)));
    }
    return pc;
}

static void test_kabsch() {
    std::printf("[kabsch]\n");
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> u(-5, 5);
    Rigid gt = make_rigid(1, -2, 0.5, 37.0, 1.5, -2.0, 0.7);
    CorrSums s;
    for (int i = 0; i < 1000; ++i) {
        double p[3] = {u(rng), u(rng), u(rng)}, q[3];
        rigid_apply(gt, p[0], p[1], p[2], q);
        s.add(p[0], p[1], p[2], q[0], q[1], q[2]);
    }
    Rigid est;
    CHECK(kabsch_from_sums(s, est), "kabsch ha fallito");
    double rot, tr;
    pose_error(est, gt, rot, tr);
    std::printf("  errore rot %.2e gradi, trasl %.2e\n", rot, tr);
    CHECK(rot < 1e-6 && tr < 1e-9, "rot=%g tr=%g", rot, tr);

    // composizione e inversa
    Rigid I = rigid_compose(rigid_inverse(gt), gt);
    CHECK(rotation_error_deg(I.R, rigid_identity().R) < 1e-6 && std::fabs(I.t[0]) + std::fabs(I.t[1]) + std::fabs(I.t[2]) < 1e-9,
          "inversa/composizione errate");
}

static void test_grid() {
    std::printf("[griglia vs brute force]\n");
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(0, 10);
    std::normal_distribution<float> g(5, 0.7f);
    PointCloud tgt;
    for (int i = 0; i < 3000; ++i) tgt.push_back(u(rng), u(rng), u(rng));      // uniformi
    for (int i = 0; i < 2000; ++i) tgt.push_back(g(rng), g(rng), g(rng));      // cluster denso

    const float dmax = 0.4f;
    Grid grid;
    grid_build(tgt, dmax, grid);
    CHECK(grid.cellStart.back() == static_cast<int>(tgt.size()), "cellStart.back()=%d", grid.cellStart.back());

    int found = 0, mismatches = 0;
    std::uniform_real_distribution<float> uq(-1, 11);  // alcune query fuori dal bounding box
    for (int i = 0; i < 4000; ++i) {
        float qx = uq(rng), qy = uq(rng), qz = uq(rng), d2g = -1, d2b = -1;
        int jg = grid_nearest(grid, qx, qy, qz, dmax, &d2g);
        int ib = brute_nearest(tgt, qx, qy, qz, dmax, &d2b);
        if ((jg < 0) != (ib < 0)) { ++mismatches; continue; }
        if (jg >= 0) {
            ++found;
            if (std::fabs(d2g - d2b) > 1e-6f) ++mismatches;
            // l'indice ordinato deve puntare allo stesso punto originale
            if (std::fabs(grid.x[jg] - tgt.x[grid.orig[jg]]) > 0) ++mismatches;
        }
    }
    std::printf("  query con vicino: %d / 4000, discordanze: %d\n", found, mismatches);
    CHECK(mismatches == 0, "%d discordanze", mismatches);
    CHECK(found > 500, "troppo poche query con vicino (%d): test poco significativo", found);
}

static void test_voxel() {
    std::printf("[voxel]\n");
    PointCloud pc;
    for (int i = 0; i < 100; ++i) pc.push_back(0.1f, 0.1f, 0.1f);  // 100 punti uguali
    pc.push_back(5.f, 5.f, 5.f);
    PointCloud out = voxel_downsample(pc, 1.0f);
    CHECK(out.size() == 2, "attesi 2 punti, trovati %zu", out.size());

    PointCloud s = make_surface(20000, 3);
    PointCloud d = voxel_downsample(s, 0.05f);
    std::printf("  superficie 20000 -> %zu punti con voxel 0.05\n", d.size());
    CHECK(d.size() < s.size() && d.size() > 100, "downsampling anomalo");
}

static void test_io() {
    std::printf("[I/O]\n");
    PointCloud pc = make_surface(1000, 5), a, b;
    std::string err;
    CHECK(save_ply("/tmp/t_io.ply", pc, &err), "%s", err.c_str());
    CHECK(load_cloud("/tmp/t_io.ply", a, &err), "%s", err.c_str());
    CHECK(a.size() == pc.size() && a.x == pc.x && a.y == pc.y && a.z == pc.z, "roundtrip PLY binario");

    {   // PLY ascii con proprieta' extra (come i modelli Stanford)
        std::ofstream f("/tmp/t_ascii.ply");
        f << "ply\nformat ascii 1.0\ncomment test\nelement vertex 3\n"
             "property float x\nproperty float y\nproperty float z\nproperty float confidence\n"
             "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
             "1.5 2.5 3.5 0.9\n-1 0 1 0.8\n0.25 0.5 0.75 1.0\n3 0 1 2\n";
    }
    PointCloud c;
    CHECK(load_cloud("/tmp/t_ascii.ply", c, &err), "%s", err.c_str());
    CHECK(c.size() == 3 && c.x[0] == 1.5f && c.z[2] == 0.75f, "lettura PLY ascii");

    {   // .bin KITTI: 3 punti x 4 float
        std::ofstream f("/tmp/t_kitti.bin", std::ios::binary);
        float v[12] = {1, 2, 3, 0.5f, 4, 5, 6, 0.5f, 7, 8, 9, 0.5f};
        f.write(reinterpret_cast<char*>(v), sizeof(v));
    }
    CHECK(load_cloud("/tmp/t_kitti.bin", b, &err), "%s", err.c_str());
    CHECK(b.size() == 3 && b.y[1] == 5.f && b.z[2] == 9.f, "lettura .bin KITTI");
    CHECK(!load_cloud("/tmp/non_esiste.ply", b, &err), "dovrebbe fallire");
}

static void test_icp() {
    std::printf("[ICP end-to-end]\n");
    // target e sorgente: campionamenti INDIPENDENTI della stessa superficie (come due scansioni)
    PointCloud tgt = make_surface(10000, 11);
    PointCloud base = make_surface(10000, 12);
    // sorgente = G^-1(base) + rumore, quindi ICP deve recuperare G (sorgente -> target)
    Rigid G = make_rigid(1, 2, 3, 6.0, 0.05, -0.03, 0.02);
    Rigid Gi = rigid_inverse(G);
    std::mt19937 rng(21);
    std::normal_distribution<double> noise(0, 0.002);
    PointCloud src;
    for (size_t i = 0; i < base.size(); ++i) {
        double o[3];
        rigid_apply(Gi, base.x[i], base.y[i], base.z[i], o);
        src.push_back(static_cast<float>(o[0] + noise(rng)), static_cast<float>(o[1] + noise(rng)),
                      static_cast<float>(o[2] + noise(rng)));
    }

    IcpParams prm;
    prm.dmax = 0.15f;
    prm.iters = 30;
    IcpResult r = icp_seq(src, tgt, prm);
    double rot, tr;
    pose_error(r.T, G, rot, tr);
    std::printf("  RMSE: %.5f -> %.5f | errore rot %.4f gradi, trasl %.5f | totale %.3f s\n", r.rmse.front(),
                r.rmse.back(), rot, tr, r.tm.total);
    std::printf("  tempi: griglia %.3f apply %.3f nn %.3f reduce %.3f solve %.3f\n", r.tm.grid, r.tm.apply,
                r.tm.nn, r.tm.reduce, r.tm.solve);
    CHECK(r.rmse.back() < r.rmse.front(), "l'RMSE non decresce");
    CHECK(rot < 0.5 && tr < 0.01, "errore troppo grande: rot=%g tr=%g", rot, tr);
}

int main() {
    test_kabsch();
    test_grid();
    test_voxel();
    test_io();
    test_icp();
    std::printf("\n%s (%d controlli falliti)\n", failures ? "TEST FALLITI" : "TUTTI I TEST OK", failures);
    return failures ? 1 : 0;
}
