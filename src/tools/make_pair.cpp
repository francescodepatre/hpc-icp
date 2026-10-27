#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <random>
#include <string>

#include "io.h"
#include "kabsch.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <input> <out_prefix> [--n N] [--rot gradi] [--trans T] [--noise S] [--outliers f] [--seed s]\n", argv[0]);
        return 1;
    }
    size_t n = 0;
    double rot_deg = 5.0, trans = 0.05, noise = 0.0, outl = 0.0;
    unsigned seed = 42;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        const char* v = argv[i + 1];
        if (a == "--n") n = static_cast<size_t>(std::atoll(v));
        else if (a == "--rot") rot_deg = std::atof(v);
        else if (a == "--trans") trans = std::atof(v);
        else if (a == "--noise") noise = std::atof(v);
        else if (a == "--outliers") outl = std::atof(v);
        else if (a == "--seed") seed = static_cast<unsigned>(std::atoi(v));
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
    }

    PointCloud in;
    std::string err;
    if (!load_cloud(argv[1], in, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    if (n == 0 || n > in.size()) n = in.size();

    std::mt19937 rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    float mn[3] = {in.x[0], in.y[0], in.z[0]}, mx[3] = {in.x[0], in.y[0], in.z[0]};
    for (size_t i = 0; i < in.size(); ++i) {
        mn[0] = std::min(mn[0], in.x[i]); mx[0] = std::max(mx[0], in.x[i]);
        mn[1] = std::min(mn[1], in.y[i]); mx[1] = std::max(mx[1], in.y[i]);
        mn[2] = std::min(mn[2], in.z[i]); mx[2] = std::max(mx[2], in.z[i]);
    }
    const double diag = std::sqrt(double(mx[0] - mn[0]) * (mx[0] - mn[0]) + double(mx[1] - mn[1]) * (mx[1] - mn[1]) +
                                  double(mx[2] - mn[2]) * (mx[2] - mn[2]));

    double ax[3] = {gauss(rng), gauss(rng), gauss(rng)};
    const double an = std::sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]);
    for (double& a : ax) a /= an;
    const double ang = rot_deg * 3.14159265358979323846 / 180.0, c = std::cos(ang), s = std::sin(ang), C = 1 - c;
    Rigid G;
    G.R[0] = c + ax[0] * ax[0] * C;           G.R[1] = ax[0] * ax[1] * C - ax[2] * s;  G.R[2] = ax[0] * ax[2] * C + ax[1] * s;
    G.R[3] = ax[1] * ax[0] * C + ax[2] * s;   G.R[4] = c + ax[1] * ax[1] * C;          G.R[5] = ax[1] * ax[2] * C - ax[0] * s;
    G.R[6] = ax[2] * ax[0] * C - ax[1] * s;   G.R[7] = ax[2] * ax[1] * C + ax[0] * s;  G.R[8] = c + ax[2] * ax[2] * C;
    double dir[3] = {gauss(rng), gauss(rng), gauss(rng)};
    const double dn = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    for (int i = 0; i < 3; ++i) G.t[i] = trans * diag * dir[i] / dn;
    const Rigid Gi = rigid_inverse(G);

    std::vector<size_t> idx(in.size());
    std::iota(idx.begin(), idx.end(), 0);
    PointCloud tgt, src;

    std::shuffle(idx.begin(), idx.end(), rng);
    for (size_t k = 0; k < n; ++k) tgt.push_back(in.x[idx[k]], in.y[idx[k]], in.z[idx[k]]);

    std::shuffle(idx.begin(), idx.end(), rng);
    for (size_t k = 0; k < n; ++k) {
        double o[3];
        rigid_apply(Gi, in.x[idx[k]], in.y[idx[k]], in.z[idx[k]], o);
        if (uni(rng) < outl) {
            o[0] = mn[0] + uni(rng) * (mx[0] - mn[0]);
            o[1] = mn[1] + uni(rng) * (mx[1] - mn[1]);
            o[2] = mn[2] + uni(rng) * (mx[2] - mn[2]);
        } else if (noise > 0) {
            for (double& v : o) v += noise * diag * gauss(rng);
        }
        src.push_back(static_cast<float>(o[0]), static_cast<float>(o[1]), static_cast<float>(o[2]));
    }

    const std::string pre = argv[2];
    if (!save_ply(pre + "_source.ply", src, &err) || !save_ply(pre + "_target.ply", tgt, &err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    std::ofstream gt(pre + "_gt.txt");
    gt.precision(17);
    for (int i = 0; i < 3; ++i) gt << G.R[3 * i] << " " << G.R[3 * i + 1] << " " << G.R[3 * i + 2] << " " << G.t[i] << "\n";
    gt << "0 0 0 1\n";
    std::printf("writed %s_source.ply, %s_target.ply (%zu points), %s_gt.txt | diagonal %.4f, rot %.2f deg, trasl %.4f\n", pre.c_str(), pre.c_str(), n, pre.c_str(), diag, rot_deg, trans * diag);
    return 0;
}
