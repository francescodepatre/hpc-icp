// Driver MPI (e ibrido MPI+OpenMP).
//
//   mpirun -np P ./icp_mpi <sorgente> <target> [--dmax D] [--iters K] [--voxel V]
//          [--gt gt.txt] [--out allineata.ply]
//          [--threads T]                      thread OpenMP per rank (default 1 = MPI puro)
//          [--partition block|cyclic]         distribuzione dei punti sorgente (default block)
//          [--collective reduce|allreduce]    come si sommano le somme (default reduce)
//
// Il rank 0 legge i file, fa il downsampling e distribuisce le nuvole con MPI_Bcast:
// i file non devono essere visibili agli altri rank e il filesystem e' letto una volta sola.
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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#include "icp.h"
#include "icp_mpi.h"
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
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ------------------------------------------------------------------ argomenti
    // Tutti i rank li leggono allo stesso modo: se c'e' un errore escono tutti insieme.
    auto usage = [&]() {
        if (rank == 0)
            std::fprintf(stderr,
                         "uso: %s <sorgente> <target> [--dmax D] [--iters K] [--voxel V] [--gt f] [--out f]\n"
                         "          [--threads T] [--partition block|cyclic] [--collective reduce|allreduce]\n",
                         argv[0]);
    };
    if (argc < 3) { usage(); MPI_Finalize(); return 1; }

    const std::string src_path = argv[1], tgt_path = argv[2];
    IcpParams prm;
    MpiOptions opt;
    float voxel = 0.f;
    int threads = 1;
    std::string gt_path, out_path;

    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (i + 1 >= argc) {
            if (rank == 0) std::fprintf(stderr, "argomento senza valore: %s\n", a.c_str());
            MPI_Finalize();
            return 1;
        }
        const std::string v = argv[++i];
        if (a == "--dmax") prm.dmax = static_cast<float>(std::atof(v.c_str()));
        else if (a == "--iters") prm.iters = std::atoi(v.c_str());
        else if (a == "--voxel") voxel = static_cast<float>(std::atof(v.c_str()));
        else if (a == "--gt") gt_path = v;
        else if (a == "--out") out_path = v;
        else if (a == "--threads") threads = std::atoi(v.c_str());
        else if (a == "--partition") {
            if (v == "block") opt.partition = Partition::Block;
            else if (v == "cyclic") opt.partition = Partition::Cyclic;
            else { usage(); MPI_Finalize(); return 1; }
        } else if (a == "--collective") {
            if (v == "reduce") opt.collective = Collective::ReduceBcast;
            else if (v == "allreduce") opt.collective = Collective::Allreduce;
            else { usage(); MPI_Finalize(); return 1; }
        } else {
            if (rank == 0) std::fprintf(stderr, "opzione sconosciuta: %s\n", a.c_str());
            MPI_Finalize();
            return 1;
        }
    }

    // Thread OpenMP per rank: di default 1, cosi' "MPI puro" non usa thread nascosti
    // (altrimenti OpenMP userebbe tutti i core in ogni rank).
#ifdef _OPENMP
    omp_set_num_threads(threads > 0 ? threads : 1);
    if (!std::getenv("OMP_SCHEDULE")) omp_set_schedule(omp_sched_dynamic, 64);  // vedi esperimento schedule
#else
    if (threads > 1 && rank == 0) std::fprintf(stderr, "avviso: compilato senza OpenMP, --threads ignorato\n");
    threads = 1;
#endif

    // ------------------------------------------------------------------ I/O sul rank 0 + Bcast
    PointCloud src, tgt;
    std::string err;
    int ok = 1;
    double t_io = 0, t_voxel = 0;
    if (rank == 0) {
        Timer tio;
        if (!load_cloud(src_path, src, &err) || !load_cloud(tgt_path, tgt, &err)) {
            std::fprintf(stderr, "errore: %s\n", err.c_str());
            ok = 0;
        }
        t_io = tio.toc();
        if (ok && voxel > 0.f) {
            Timer tv;
            src = voxel_downsample(src, voxel);
            tgt = voxel_downsample(tgt, voxel);
            t_voxel = tv.toc();
        }
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!ok) { MPI_Finalize(); return 1; }

    Timer tb;
    bcast_cloud(src, 0);
    bcast_cloud(tgt, 0);
    const double t_bcast = tb.toc();

    if (rank == 0) {
        std::printf("rank MPI: %d, thread OpenMP per rank: %d, partizione: %s, collettiva: %s\n", size, threads,
                    opt.partition == Partition::Block ? "block" : "cyclic",
                    opt.collective == Collective::ReduceBcast ? "reduce+bcast" : "allreduce");
        std::printf("sorgente %zu punti, target %zu punti (lettura %.3f s, voxel %.3f s, bcast %.3f s)\n",
                    src.size(), tgt.size(), t_io, t_voxel, t_bcast);
    }

    // ------------------------------------------------------------------ ICP
    // Tutti i rank chiamano icp_mpi (mai dentro un if (rank == 0)).
    MpiStats st;
    IcpResult r = icp_mpi(src, tgt, prm, rigid_identity(), opt, &st);

    // Bilanciamento: tempo di calcolo locale massimo / medio tra i rank (1.0 = perfetto)
    double cmax = 0, csum = 0;
    MPI_Reduce(&st.compute, &cmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&st.compute, &csum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double imbalance = csum > 0 ? cmax / (csum / size) : 1.0;

        std::printf("\nICP MPI: dmax=%.4f iters=%d\n", prm.dmax, prm.iters);
        if (!r.rmse.empty())
            std::printf("RMSE prima iterazione: %.6f  ultima: %.6f  (coppie valide ultima: %zu)\n", r.rmse.front(),
                        r.rmse.back(), r.matches.back());
        std::printf("T stimata (sorgente -> target):\n");
        for (int i = 0; i < 3; ++i)
            std::printf("  % .6f % .6f % .6f % .6f\n", r.T.R[3 * i], r.T.R[3 * i + 1], r.T.R[3 * i + 2], r.T.t[i]);

        double rot = -1, tr = -1;
        if (!gt_path.empty()) {
            Rigid gt;
            if (!load_pose(gt_path, gt)) {
                std::fprintf(stderr, "impossibile leggere il ground truth %s\n", gt_path.c_str());
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            pose_error(r.T, gt, rot, tr);
            std::printf("errore rotazione: %.5f gradi, errore traslazione: %.6f\n", rot, tr);
        }

        const IcpTimings& tm = r.tm;
        std::printf("\ntempi rank 0 (s): griglia %.4f | apply %.4f | nn %.4f | somme locali %.4f | solve %.4f | "
                    "comunicazione %.4f | totale %.4f\n",
                    tm.grid, tm.apply, tm.nn, tm.reduce, tm.solve, st.comm, tm.total);
        std::printf("sbilanciamento del calcolo (max/medio tra i rank): %.3f\n", imbalance);

        // Riga CSV: stessi campi di icp_seq/icp_omp fino a 'trans', poi np, thread, comm, sbilanciamento.
        // 'reduce' qui sono le somme LOCALI; la comunicazione e' nel campo 'comm'.
        std::printf("CSV,mpi,%zu,%zu,%d,%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d,%.6f,%.4f\n", src.size(),
                    tgt.size(), prm.iters, prm.dmax, tm.grid, tm.apply, tm.nn, tm.reduce, tm.solve, tm.total, rot, tr,
                    size, threads, st.comm, imbalance);

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
            if (!save_ply(out_path, aligned, &err)) {
                std::fprintf(stderr, "%s\n", err.c_str());
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            std::printf("sorgente allineata salvata in %s\n", out_path.c_str());
        }
    }

    MPI_Finalize();
    return 0;
}
