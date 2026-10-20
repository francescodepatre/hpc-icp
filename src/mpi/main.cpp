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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "icp.h"
#include "icp_mpi.h"
#include "io.h"
#include "metrics.h"
#include "voxel.h"

namespace fs = std::filesystem;
using Mat4 = std::array<double, 16>;


// ============================================================
// Matrici omogenee 4x4
// ============================================================

static Mat4 identity4()
{
    Mat4 M{};

    M[0] = 1.0;
    M[5] = 1.0;
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


// ============================================================
// Lettura pose KITTI
// ============================================================

static bool load_kitti_poses(
    const fs::path& path,
    std::vector<Mat4>& poses)
{
    std::ifstream f(path);

    if (!f.is_open()) {
        std::cerr << "Impossibile aprire le pose: "
                  << path << '\n';
        return false;
    }

    poses.clear();

    std::string line;

    while (std::getline(f, line)) {
        if (line.find_first_not_of(" \t\r\n")
            == std::string::npos) {
            continue;
        }

        std::istringstream iss(line);
        double m[12];

        for (int i = 0; i < 12; ++i) {
            if (!(iss >> m[i])) {
                std::cerr << "Riga pose non valida: "
                          << path << '\n';
                return false;
            }
        }

        poses.push_back(to4x4(m));
    }

    return !poses.empty();
}


// ============================================================
// Lettura calibrazione KITTI
// ============================================================

static bool load_kitti_calib(
    const fs::path& path,
    Mat4& Tr)
{
    std::ifstream f(path);

    if (!f.is_open()) {
        std::cerr << "Impossibile aprire la calibrazione: "
                  << path << '\n';
        return false;
    }

    std::string line;

    while (std::getline(f, line)) {
        const auto first = line.find_first_not_of(" \t");

        if (first == std::string::npos) {
            continue;
        }

        const std::string trimmed = line.substr(first);

        if (trimmed.rfind("Tr:", 0) != 0) {
            continue;
        }

        std::istringstream iss(trimmed.substr(3));
        double m[12];

        for (int i = 0; i < 12; ++i) {
            if (!(iss >> m[i])) {
                std::cerr << "Matrice Tr non valida: "
                          << path << '\n';
                return false;
            }
        }

        Tr = to4x4(m);
        return true;
    }

    std::cerr << "Matrice Tr non trovata in: "
              << path << '\n';

    return false;
}


// ============================================================
// Ground truth source -> target
// ============================================================

static Rigid compute_ground_truth(
    const std::vector<Mat4>& poses,
    const Mat4& Tr,
    int src_idx,
    int tgt_idx)
{
    if (src_idx < 0 || tgt_idx < 0 ||
        static_cast<size_t>(src_idx) >= poses.size() ||
        static_cast<size_t>(tgt_idx) >= poses.size()) {
        throw std::out_of_range(
            "Indici non validi per il ground truth");
    }

    const Mat4 Tr_inv = inverse_rigid4(Tr);
    const Mat4 P_tgt_inv = inverse_rigid4(poses[tgt_idx]);

    Mat4 T = multiply4(Tr_inv, P_tgt_inv);
    T = multiply4(T, poses[src_idx]);
    T = multiply4(T, Tr);

    return to_rigid(T);
}


// ============================================================
// Gestione scansioni e sequenze
// ============================================================

static bool is_bin_file(const fs::path& p)
{
    return p.extension() == ".bin";
}


static bool contains_bin(const fs::path& dir)
{
    if (!fs::is_directory(dir)) {
        return false;
    }

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && is_bin_file(entry.path())) {
            return true;
        }
    }

    return false;
}


static std::vector<fs::path> list_scans(const fs::path& sequence_dir)
{
    fs::path scan_dir = sequence_dir;

    if (!contains_bin(scan_dir) &&
        fs::is_directory(sequence_dir / "velodyne")) {
        scan_dir = sequence_dir / "velodyne";
    }

    std::vector<fs::path> scans;

    for (const auto& entry : fs::directory_iterator(scan_dir)) {
        if (entry.is_regular_file() && is_bin_file(entry.path())) {
            scans.push_back(entry.path());
        }
    }

    std::sort(
        scans.begin(),
        scans.end(),
        [](const fs::path& a, const fs::path& b) {
            return a.filename().string() < b.filename().string();
        });

    return scans;
}


static int scan_index(const fs::path& path)
{
    return std::stoi(path.stem().string());
}


static std::string sequence_name(const fs::path& path)
{
    if (path.filename() == "velodyne") {
        return path.parent_path().filename().string();
    }

    return path.filename().string();
}


// Risolve poses/00.txt, poses/00 oppure poses/.
static fs::path resolve_pose_path(
    const fs::path& poses_arg,
    const std::string& sequence)
{
    if (fs::is_regular_file(poses_arg)) {
        return poses_arg;
    }

    const fs::path with_extension =
        poses_arg.string() + ".txt";

    if (!fs::exists(poses_arg) &&
        fs::is_regular_file(with_extension)) {
        return with_extension;
    }

    return poses_arg / (sequence + ".txt");
}


// Risolve sia la directory della sequenza sia quella generale.
static fs::path resolve_calib_path(
    const fs::path& calib_arg,
    const std::string& sequence)
{
    if (fs::is_regular_file(calib_arg / "calib.txt")) {
        return calib_arg / "calib.txt";
    }

    return calib_arg / sequence / "calib.txt";
}


// ============================================================
// Main
// ============================================================

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto usage = [&]() {
        if (rank == 0) {
            std::cerr
                << "Uso:\n"
                << argv[0]
                << " <scansioni_dir> <risultati.csv>"
                << " <poses_file_o_dir> <calib_dir_o_sequenza>\n"
                << " [--dmax D] [--iters K] [--voxel V]"
                << " [--threads T] [--out DIR]\n"
                << " [--partition block|cyclic]"
                << " [--collective reduce|allreduce]\n";
        }
    };

    if (argc < 5) {
        usage();
        MPI_Finalize();
        return 1;
    }

    const fs::path scans_root = argv[1];
    const fs::path csv_path = argv[2];
    const fs::path poses_arg = argv[3];
    const fs::path calib_arg = argv[4];

    IcpParams prm;
    MpiOptions opt;

    float voxel = 0.0f;
    int threads = 1;

    fs::path out_dir;

    for (int i = 5; i < argc; ++i) {
        const std::string a = argv[i];

        if (i + 1 >= argc) {
            if (rank == 0) {
                std::cerr << "Argomento senza valore: "
                          << a << '\n';
            }

            MPI_Finalize();
            return 1;
        }

        const std::string v = argv[++i];

        if (a == "--dmax") {
            prm.dmax = std::strtof(v.c_str(), nullptr);
        }
        else if (a == "--iters") {
            prm.iters = std::atoi(v.c_str());
        }
        else if (a == "--voxel") {
            voxel = std::strtof(v.c_str(), nullptr);
        }
        else if (a == "--threads") {
            threads = std::atoi(v.c_str());
        }
        else if (a == "--out") {
            out_dir = v;
        }
        else if (a == "--partition") {
            if (v == "block") {
                opt.partition = Partition::Block;
            }
            else if (v == "cyclic") {
                opt.partition = Partition::Cyclic;
            }
            else {
                usage();
                MPI_Finalize();
                return 1;
            }
        }
        else if (a == "--collective") {
            if (v == "reduce") {
                opt.collective = Collective::ReduceBcast;
            }
            else if (v == "allreduce") {
                opt.collective = Collective::Allreduce;
            }
            else {
                usage();
                MPI_Finalize();
                return 1;
            }
        }
        else {
            if (rank == 0) {
                std::cerr << "Opzione sconosciuta: "
                          << a << '\n';
            }

            MPI_Finalize();
            return 1;
        }
    }

    if (prm.iters <= 0 || prm.dmax <= 0.0f ||
        voxel < 0.0f || threads <= 0) {
        if (rank == 0) {
            std::cerr << "Parametri non validi.\n";
        }

        MPI_Finalize();
        return 1;
    }

#ifdef _OPENMP
    omp_set_num_threads(threads);

    if (!std::getenv("OMP_SCHEDULE")) {
        omp_set_schedule(omp_sched_dynamic, 64);
    }
#else
    if (threads > 1 && rank == 0) {
        std::cerr
            << "Avviso: compilato senza OpenMP; "
            << "--threads ignorato.\n";
    }

    threads = 1;
#endif

    // --------------------------------------------------------
    // Preparazione directory e CSV: solo rank 0
    // --------------------------------------------------------

    int setup_ok = 1;
    std::ofstream csv;

    if (rank == 0) {
        try {
            if (!fs::is_directory(scans_root)) {
                throw std::runtime_error(
                    "Directory scansioni non valida: " +
                    scans_root.string());
            }

            if (!csv_path.parent_path().empty()) {
                fs::create_directories(csv_path.parent_path());
            }

            if (!out_dir.empty()) {
                fs::create_directories(out_dir);
            }

            csv.open(csv_path);

            if (!csv.is_open()) {
                throw std::runtime_error(
                    "Impossibile creare il CSV: " +
                    csv_path.string());
            }

            csv
                << "sequence,src_idx,tgt_idx,"
                << "n_src,n_tgt,iters,dmax,voxel,"
                << "mpi_ranks,threads,"
                << "t_grid,t_apply,t_nn,t_reduce,t_solve,"
                << "t_total,t_comm,rot_err,trans_err\n";

            std::cout << "Rank MPI: " << size
                      << ", thread OpenMP per rank: "
                      << threads << '\n';

            std::cout << "Directory scansioni: "
                      << scans_root << '\n';

            std::cout << "CSV: " << csv_path << "\n\n";
        }
        catch (const std::exception& e) {
            std::cerr << "Errore inizializzazione: "
                      << e.what() << '\n';
            setup_ok = 0;
        }
    }

    MPI_Bcast(&setup_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!setup_ok) {
        MPI_Finalize();
        return 1;
    }

    // --------------------------------------------------------
    // Individua le sequenze oppure usa la directory passata
    // direttamente, se contiene già le scansioni .bin.
    // --------------------------------------------------------

    std::vector<fs::path> sequence_dirs;

    try {
        if (contains_bin(scans_root) ||
            fs::is_directory(scans_root / "velodyne")) {
            sequence_dirs.push_back(scans_root);
        }
        else {
            for (const auto& entry :
                 fs::directory_iterator(scans_root)) {
                if (entry.is_directory()) {
                    sequence_dirs.push_back(entry.path());
                }
            }

            std::sort(
                sequence_dirs.begin(),
                sequence_dirs.end(),
                [](const fs::path& a, const fs::path& b) {
                    return a.filename().string() <
                           b.filename().string();
                });
        }
    }
    catch (const fs::filesystem_error& e) {
        if (rank == 0) {
            std::cerr << "Errore ricerca sequenze: "
                      << e.what() << '\n';
        }

        MPI_Finalize();
        return 1;
    }

    size_t total_pairs = 0;
    size_t failed_pairs = 0;

    // ========================================================
    // Ciclo sulle sequenze
    // ========================================================

    for (const fs::path& sequence_dir : sequence_dirs) {
        const std::string sequence =
            sequence_name(sequence_dir);

        std::vector<fs::path> scans;

        try {
            scans = list_scans(sequence_dir);
        }
        catch (const fs::filesystem_error& e) {
            if (rank == 0) {
                std::cerr << "Errore lettura scansioni: "
                          << e.what() << '\n';
            }

            continue;
        }

        if (scans.size() < 2) {
            if (rank == 0) {
                std::cerr << "Meno di due scansioni in "
                          << sequence_dir << '\n';
            }

            continue;
        }

        if (rank == 0) {
            std::cout
                << "\n========================================\n"
                << "Sequenza " << sequence << '\n'
                << "Scansioni trovate: " << scans.size() << '\n'
                << "========================================\n";
        }

        // Pose e calibrazione vengono caricate solo dal rank 0.
        std::vector<Mat4> poses;
        Mat4 Tr = identity4();
        bool gt_available = false;

        if (rank == 0) {
            const fs::path pose_path =
                resolve_pose_path(poses_arg, sequence);

            const fs::path calib_path =
                resolve_calib_path(calib_arg, sequence);

            std::cout << "File pose: " << pose_path << '\n';
            std::cout << "File calibrazione: "
                      << calib_path << '\n';

            const bool poses_ok =
                load_kitti_poses(pose_path, poses);

            const bool calib_ok =
                load_kitti_calib(calib_path, Tr);

            gt_available = poses_ok && calib_ok;

            if (!gt_available) {
                std::cerr
                    << "Ground truth non disponibile per "
                    << sequence
                    << "; l'ICP continuerà senza errori GT.\n";
            }
        }

        // ====================================================
        // Ciclo sulle coppie consecutive
        // source = scans[i], target = scans[i-1]
        // ====================================================

        for (size_t i = 1; i < scans.size(); ++i) {
            const fs::path src_path = scans[i];
            const fs::path tgt_path = scans[i - 1];

            int src_idx = -1;
            int tgt_idx = -1;

            try {
                src_idx = scan_index(src_path);
                tgt_idx = scan_index(tgt_path);
            }
            catch (const std::exception& e) {
                if (rank == 0) {
                    std::cerr << "Nome scansione non valido: "
                              << e.what() << '\n';
                }

                ++failed_pairs;
                continue;
            }

            if (rank == 0) {
                std::cout << "\n[" << sequence << "] "
                          << src_path.filename().string()
                          << " -> "
                          << tgt_path.filename().string()
                          << '\n';
            }

            PointCloud src, tgt;
            std::string err;
            int pair_ok = 1;

            // Il rank 0 legge le scansioni e applica il voxel.
            if (rank == 0) {
                if (!load_cloud(src_path.string(), src, &err)) {
                    std::cerr << "Errore source: "
                              << err << '\n';
                    pair_ok = 0;
                }

                if (pair_ok &&
                    !load_cloud(tgt_path.string(), tgt, &err)) {
                    std::cerr << "Errore target: "
                              << err << '\n';
                    pair_ok = 0;
                }

                if (pair_ok && (src.empty() || tgt.empty())) {
                    std::cerr << "Nuvola vuota.\n";
                    pair_ok = 0;
                }

                if (pair_ok && voxel > 0.0f) {
                    src = voxel_downsample(src, voxel);
                    tgt = voxel_downsample(tgt, voxel);
                }

                if (pair_ok && (src.empty() || tgt.empty())) {
                    std::cerr
                        << "Nuvola vuota dopo voxel downsampling.\n";
                    pair_ok = 0;
                }
            }

            // Tutti i rank devono sapere se la coppia è valida.
            MPI_Bcast(&pair_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

            if (!pair_ok) {
                ++failed_pairs;
                continue;
            }

            // Distribuisce le due nuvole a tutti i processi.
            const double bcast_start = MPI_Wtime();

            bcast_cloud(src, 0);
            bcast_cloud(tgt, 0);

            const double t_bcast = MPI_Wtime() - bcast_start;

            if (rank == 0) {
                std::cout << "Punti source: " << src.size()
                          << ", target: " << tgt.size()
                          << ", broadcast: " << t_bcast
                          << " s\n";
            }

            // ------------------------------------------------
            // ICP MPI
            // ------------------------------------------------

            MpiStats st;
            IcpResult r;
            int icp_ok = 1;
            double wall_total = 0.0;

            try {
                Timer total_timer;

                r = icp_mpi(
                    src,
                    tgt,
                    prm,
                    rigid_identity(),
                    opt,
                    &st);

                wall_total = total_timer.toc();

                if (r.rmse.empty() || r.matches.empty()) {
                    throw std::runtime_error(
                        "ICP non ha prodotto risultati validi");
                }
            }
            catch (const std::exception& e) {
                std::cerr << "[rank " << rank
                          << "] Errore ICP: "
                          << e.what() << '\n';

                icp_ok = 0;
            }

            // In caso di eccezione, tutti i rank devono uscire
            // dal ciclo della coppia nello stesso modo.
            int all_ok = 0;

            MPI_Allreduce(
                &icp_ok, &all_ok, 1,
                MPI_INT, MPI_MIN, MPI_COMM_WORLD);

            if (!all_ok) {
                ++failed_pairs;
                continue;
            }

            // ------------------------------------------------
            // Metriche e CSV: solo rank 0
            // ------------------------------------------------

            if (rank == 0) {
                std::cout
                    << "RMSE iniziale: " << r.rmse.front()
                    << ", finale: " << r.rmse.back()
                    << ", corrispondenze: "
                    << r.matches.back() << '\n';

                std::cout << "T stimata (source -> target):\n";

                for (int row = 0; row < 3; ++row) {
                    std::cout
                        << "  "
                        << std::fixed << std::setprecision(6)
                        << r.T.R[3 * row] << " "
                        << r.T.R[3 * row + 1] << " "
                        << r.T.R[3 * row + 2] << " "
                        << r.T.t[row] << '\n';
                }

                double rot_err = -1.0;
                double trans_err = -1.0;

                if (gt_available &&
                    src_idx >= 0 && tgt_idx >= 0 &&
                    static_cast<size_t>(src_idx) < poses.size() &&
                    static_cast<size_t>(tgt_idx) < poses.size()) {

                    const Rigid gt = compute_ground_truth(
                        poses, Tr, src_idx, tgt_idx);

                    pose_error(
                        r.T, gt, rot_err, trans_err);

                    std::cout
                        << "Errore rotazione: "
                        << rot_err << " gradi, "
                        << "errore traslazione: "
                        << trans_err << " m\n";
                }
                else {
                    std::cout
                        << "Ground truth non disponibile "
                        << "per questa coppia.\n";
                }

                const IcpTimings& tm = r.tm;

                std::cout
                    << "Tempi rank 0 (s): griglia "
                    << tm.grid
                    << " | apply " << tm.apply
                    << " | NN " << tm.nn
                    << " | somme locali " << tm.reduce
                    << " | solve " << tm.solve
                    << " | comunicazione MPI " << st.comm
                    << " | totale ICP " << tm.total
                    << " | totale misurato " << wall_total
                    << '\n';

                csv << sequence << ','
                    << src_idx << ','
                    << tgt_idx << ','
                    << src.size() << ','
                    << tgt.size() << ','
                    << prm.iters << ','
                    << prm.dmax << ','
                    << voxel << ','
                    << size << ','
                    << threads << ','
                    << tm.grid << ','
                    << tm.apply << ','
                    << tm.nn << ','
                    << tm.reduce << ','
                    << tm.solve << ','
                    << wall_total << ','
                    << st.comm << ','
                    << rot_err << ','
                    << trans_err << '\n';

                csv.flush();

                // Salvataggio opzionale della nuvola allineata.
                if (!out_dir.empty()) {
                    PointCloud aligned;
                    aligned.resize(src.size());

                    for (size_t k = 0; k < src.size(); ++k) {
                        double o[3];

                        rigid_apply(
                            r.T,
                            src.x[k],
                            src.y[k],
                            src.z[k],
                            o);

                        aligned.x[k] =
                            static_cast<float>(o[0]);

                        aligned.y[k] =
                            static_cast<float>(o[1]);

                        aligned.z[k] =
                            static_cast<float>(o[2]);
                    }

                    std::ostringstream name;

                    name << sequence << "_"
                         << std::setw(6) << std::setfill('0')
                         << src_idx << "_to_"
                         << std::setw(6) << std::setfill('0')
                         << tgt_idx << ".ply";

                    const fs::path ply_path =
                        out_dir / name.str();

                    if (!save_ply(
                            ply_path.string(), aligned, &err)) {
                        std::cerr
                            << "Errore salvataggio PLY: "
                            << err << '\n';
                    }
                }
            }

            ++total_pairs;
        }
    }

    if (rank == 0) {
        csv.close();

        std::cout
            << "\n========================================\n"
            << "Benchmark MPI terminato\n"
            << "Coppie completate: " << total_pairs << '\n'
            << "Coppie fallite/saltate: " << failed_pairs << '\n'
            << "CSV salvato in: " << csv_path << '\n'
            << "========================================\n";
    }

    // Riunisce i contatori, che sono locali a ciascun processo.
    unsigned long long local_counts[2] = {
        static_cast<unsigned long long>(total_pairs),
        static_cast<unsigned long long>(failed_pairs)
    };

    unsigned long long global_counts[2] = {0, 0};

    MPI_Reduce(
        local_counts, global_counts, 2,
        MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}