// Driver della versione sequenziale.
//
// Uso:
//   icp_seq <sorgente> <target> [--dmax D] [--iters K] [--voxel V]
//           [--gt gt.txt] [--out allineata.ply]
//
//  sorgente/target : file .ply oppure .bin (KITTI Velodyne)
//  --dmax          : distanza massima di corrispondenza (default 0.5)
//  --iters         : numero fisso di iterazioni (default 30)
//  --voxel         : lato del voxel per il downsampling di entrambe le nuvole (default: nessuno)
//  --gt            : file di testo con 16 numeri (matrice 4x4 per righe) = trasformazione
//                    vera sorgente -> target, per calcolare gli errori
//  --out           : salva la sorgente allineata in un PLY
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "icp.h"
#include "io.h"
#include "metrics.h"
#include "voxel.h"

static bool load_pose(const std::string& path, Rigid& T) {
    std::ifstream f(path);
    double m[16];
    for (int i = 0; i < 16; ++i)
        if (!(f >> m[i])) return false;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) T.R[3 * i + j] = m[4 * i + j];
        T.t[i] = m[4 * i + 3];
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "uso: %s <sorgente> <target> [--dmax D] [--iters K] [--voxel V] [--gt f] [--out f]\n", argv[0]);
        return 1;
    }
    const std::string src_path = argv[1], tgt_path = argv[2];
    IcpParams prm;
    float voxel = 0.f;
    std::string gt_path, out_path;

    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (i + 1 >= argc) { std::fprintf(stderr, "argomento senza valore: %s\n", a.c_str()); return 1; }
        const char* v = argv[++i];
        if (a == "--dmax") prm.dmax = static_cast<float>(std::atof(v));
        else if (a == "--iters") prm.iters = std::atoi(v);
        else if (a == "--voxel") voxel = static_cast<float>(std::atof(v));
        else if (a == "--gt") gt_path = v;
        else if (a == "--out") out_path = v;
        else { std::fprintf(stderr, "opzione sconosciuta: %s\n", a.c_str()); return 1; }
    }

    std::string err;
    PointCloud src, tgt;
    Timer tio;
    if (!load_cloud(src_path, src, &err) || !load_cloud(tgt_path, tgt, &err)) {
        std::fprintf(stderr, "errore: %s\n", err.c_str());
        return 1;
    }
    const double t_io = tio.toc();
    std::printf("caricati: sorgente %zu punti, target %zu punti (%.3f s)\n", src.size(), tgt.size(), t_io);

    if (voxel > 0.f) {
        Timer tv;
        src = voxel_downsample(src, voxel);
        tgt = voxel_downsample(tgt, voxel);
        std::printf("voxel %.4f: sorgente %zu, target %zu (%.3f s)\n", voxel, src.size(), tgt.size(), tv.toc());
    }

    IcpResult r = icp_seq(src, tgt, prm);

    std::printf("\nICP: dmax=%.4f iters=%d\n", prm.dmax, prm.iters);
    std::printf("RMSE prima iterazione: %.6f  ultima: %.6f  (coppie valide ultima: %zu)\n",
                r.rmse.front(), r.rmse.back(), r.matches.back());
    std::printf("T stimata (sorgente -> target):\n");
    for (int i = 0; i < 3; ++i)
        std::printf("  % .6f % .6f % .6f % .6f\n", r.T.R[3 * i], r.T.R[3 * i + 1], r.T.R[3 * i + 2], r.T.t[i]);

    double rot = -1, tr = -1;
    if (!gt_path.empty()) {
        Rigid gt;
        if (!load_pose(gt_path, gt)) {
            std::fprintf(stderr, "impossibile leggere il ground truth %s\n", gt_path.c_str());
            return 1;
        }
        pose_error(r.T, gt, rot, tr);
        std::printf("errore rotazione: %.5f gradi, errore traslazione: %.6f\n", rot, tr);
    }

    const IcpTimings& tm = r.tm;
    std::printf("\ntempi (s): griglia %.4f | apply %.4f | nn %.4f | reduce %.4f | solve %.4f | totale %.4f\n",
                tm.grid, tm.apply, tm.nn, tm.reduce, tm.solve, tm.total);

    // Riga CSV per gli script di benchmark
    std::printf("CSV,seq,%zu,%zu,%d,%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", src.size(), tgt.size(),
                prm.iters, prm.dmax, tm.grid, tm.apply, tm.nn, tm.reduce, tm.solve, tm.total, rot, tr);

    if (!out_path.empty()) {
        PointCloud aligned;
        aligned.resize(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            double o[3];
            rigid_apply(r.T, src.x[i], src.y[i], src.z[i], o);
            aligned.x[i] = static_cast<float>(o[0]);
            aligned.y[i] = static_cast<float>(o[1]);
            aligned.z[i] = static_cast<float>(o[2]);
        }
        if (!save_ply(out_path, aligned, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        std::printf("sorgente allineata salvata in %s\n", out_path.c_str());
    }
    return 0;
}
