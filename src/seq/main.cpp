#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include <stdexcept>

#include "icp.h"
#include "io.h"
#include "metrics.h"
#include "voxel.h"

#include <filesystem>
#include <algorithm>
#include <iomanip>

#include <array>
#include <sstream>
#include <vector>

using Mat4 = std::array<double, 16>;

// Crea una matrice identità 4x4.
static Mat4 identity4()
{
    Mat4 M{};

    M[0]  = 1.0;
    M[5]  = 1.0;
    M[10] = 1.0;
    M[15] = 1.0;

    return M;
}


// Converte i 12 valori di una matrice KITTI 3x4
// in una matrice omogenea 4x4.
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


// Legge tutte le pose dal file KITTI poses/XX.txt.
// Ogni riga contiene 12 valori, corrispondenti
// a una matrice di trasformazione 3x4.
static bool load_kitti_poses(
    const std::string& path,
    std::vector<Mat4>& poses)
{
    std::ifstream f(path);

    if (!f.is_open()) {
        std::cerr << "Impossibile aprire il file delle pose: "
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
                std::cerr << "Riga non valida nel file delle pose: "
                          << path << '\n';
                return false;
            }
        }

        poses.push_back(to4x4(m));
    }

    return !poses.empty();
}


// Legge la matrice Tr dal file calib.txt di KITTI.
// Gli altri parametri di calibrazione vengono ignorati.
static bool load_kitti_calib(
    const std::string& path,
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
        if (line.rfind("Tr:", 0) != 0) {
            continue;
        }

        std::istringstream iss(line.substr(3));

        double m[12];

        for (int i = 0; i < 12; ++i) {
            if (!(iss >> m[i])) {
                std::cerr << "Matrice Tr non valida in: "
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


// Moltiplicazione di due matrici 4x4.
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


// Inversa di una trasformazione rigida:
// [R t]^-1 = [R^T  -R^T t]
// [0 1]      [0       1  ]
static Mat4 inverse_rigid4(const Mat4& T)
{
    Mat4 inv = identity4();

    // Trasposta della rotazione.
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            inv[4 * i + j] = T[4 * j + i];
        }
    }

    // Nuova traslazione: -R^T * t.
    for (int i = 0; i < 3; ++i) {
        inv[4 * i + 3] = 0.0;

        for (int j = 0; j < 3; ++j) {
            inv[4 * i + 3] -=
                inv[4 * i + j] * T[4 * j + 3];
        }
    }

    return inv;
}


// Converte una matrice 4x4 nella struttura Rigid
// già utilizzata dal tuo ICP.
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


// Calcola il ground truth per una coppia di scansioni.
//
// src_idx: indice della scansione sorgente.
// tgt_idx: indice della scansione target.
//
// La trasformazione restituita porta i punti
// dal sistema LiDAR della sorgente a quello del target.
static Rigid compute_ground_truth(
    const std::vector<Mat4>& poses,
    const Mat4& Tr,
    int src_idx,
    int tgt_idx)
{
    if (src_idx < 0 || tgt_idx < 0 ||
        src_idx >= static_cast<int>(poses.size()) ||
        tgt_idx >= static_cast<int>(poses.size())) {
        throw std::out_of_range(
            "Indice delle pose fuori dai limiti"
        );
    }

    const Mat4& P_src = poses[src_idx];
    const Mat4& P_tgt = poses[tgt_idx];

    const Mat4 Tr_inv = inverse_rigid4(Tr);
    const Mat4 P_tgt_inv = inverse_rigid4(P_tgt);

    // T_GT = Tr^-1 * P_tgt^-1 * P_src * Tr
    Mat4 T = multiply4(
        multiply4(
            multiply4(Tr_inv, P_tgt_inv),
            P_src
        ),
        Tr
    );

    return to_rigid(T);
}

int analyze_sequence(
    const std::string& src_path,
    const std::string& tgt_path,
    float voxel,
    const IcpParams& prm,
    const Rigid* gt,
    const std::string& out_path,
    std::ofstream& csv,
    const std::string& sequence,
    int src_idx,
    int tgt_idx
)
{
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

    double rot = -1.0;
    double tr = -1.0;

    if (gt != nullptr) {
        pose_error(r.T, *gt, rot, tr);

        std::printf(
            "Errore rotazione: %.5f gradi, "
            "errore traslazione: %.6f\n",
            rot, tr
        );
    } else {
        std::printf(
            "Ground truth non disponibile: errori non calcolati.\n"
        );
    }

    const IcpTimings& tm = r.tm;
    std::printf("\ntempi (s): griglia %.4f | apply %.4f | nn %.4f | reduce %.4f | solve %.4f | totale %.4f\n",
                tm.grid, tm.apply, tm.nn, tm.reduce, tm.solve, tm.total);

    csv << sequence << ',' << src_idx << ',' << tgt_idx << ',' << src.size() << ',' << tgt.size() << ',' << prm.iters << ',' << prm.dmax << ',' << tm.grid << ',' << tm.apply << ',' << tm.nn << ',' << tm.reduce << ',' << tm.solve << ',' << tm.total << ',' << rot << ',' << tr << '\n';
    csv.flush();

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

namespace fs = std::filesystem;

int main(int argc, char** argv)
{
    // Uso:
    // ./icp_seq <scansioni_dir> <risultati.csv>
    //           <pose_file> <calib_dir>
    //           [--dmax D] [--iters K] [--voxel V] [--out FILE]

    if (argc < 5) {
        std::fprintf(
            stderr,
            "Uso: %s <scansioni_dir> <risultati.csv> "
            "<pose_file> <calib_dir> "
            "[--dmax D] [--iters K] [--voxel V] [--out FILE]\n",
            argv[0]
        );
        return 1;
    }

    // Percorsi passati da riga di comando.
    const fs::path scan_dir = argv[1];
    const fs::path csv_path = argv[2];
    const fs::path pose_file = argv[3];
    const fs::path calib_dir = argv[4];

    IcpParams prm;
    float voxel = 0.0f;
    std::string out_path;

    // Lettura delle opzioni.
    for (int i = 5; i < argc; ++i) {
        const std::string a = argv[i];

        if (i + 1 >= argc) {
            std::cerr << "Argomento senza valore: " << a << '\n';
            return 1;
        }

        const char* v = argv[++i];

        if (a == "--dmax") {
            prm.dmax = static_cast<float>(std::atof(v));
        }
        else if (a == "--iters") {
            prm.iters = std::atoi(v);
        }
        else if (a == "--voxel") {
            voxel = static_cast<float>(std::atof(v));
        }
        else if (a == "--out") {
            out_path = v;
        }
        else {
            std::cerr << "Opzione sconosciuta: " << a << '\n';
            return 1;
        }
    }

    // --------------------------------------------------
    // 1. Controllo dei percorsi
    // --------------------------------------------------

    if (!fs::is_directory(scan_dir)) {
        std::cerr << "Directory delle scansioni non valida: "
                  << scan_dir << '\n';
        return 1;
    }

    if (!fs::is_regular_file(pose_file)) {
        std::cerr << "File delle pose non trovato: "
                  << pose_file << '\n';
        return 1;
    }

    if (!fs::is_directory(calib_dir)) {
        std::cerr << "Directory della calibrazione non valida: "
                  << calib_dir << '\n';
        return 1;
    }

    // Ricava il nome della sequenza dalla directory.
    // Esempio: .../sequences/00 -> "00"
    const std::string sequence = scan_dir.filename().string();

    std::cout << "\n====================================\n"
              << "Sequenza selezionata: " << sequence << '\n'
              << "Directory scansioni: " << scan_dir << '\n'
              << "File pose: " << pose_file << '\n'
              << "Directory calibrazione: " << calib_dir << '\n'
              << "====================================\n";

    // --------------------------------------------------
    // 2. Caricamento delle scansioni della sola sequenza
    // --------------------------------------------------

    std::vector<fs::path> scans;

    for (const auto& entry : fs::directory_iterator(scan_dir)) {
        if (entry.is_regular_file() &&
            entry.path().extension() == ".bin") {
            scans.push_back(entry.path());
        }
    }

    // Ordine numerico dei nomi: 000000.bin, 000001.bin, ...
    std::sort(
        scans.begin(),
        scans.end(),
        [](const fs::path& a, const fs::path& b) {
            return std::stoi(a.stem().string()) <
                   std::stoi(b.stem().string());
        }
    );

    if (scans.size() < 2) {
        std::cerr << "Meno di due file .bin trovati in "
                  << scan_dir << '\n';
        return 1;
    }

    std::cout << "Scansioni trovate: " << scans.size() << '\n';

    // --------------------------------------------------
    // 3. Caricamento del ground truth
    // --------------------------------------------------

    // Accetta sia:
    //   calib_dir = .../sequences/00
    // sia:
    //   calib_dir = .../sequences/00/
    //
    // Il file atteso è calib.txt dentro questa directory.
    const fs::path calib_file = calib_dir / "calib.txt";

    std::vector<Mat4> poses;
    Mat4 Tr{};

    const bool poses_ok =
        load_kitti_poses(pose_file.string(), poses);

    const bool calib_ok =
        load_kitti_calib(calib_file.string(), Tr);

    const bool have_gt = poses_ok && calib_ok;

    if (have_gt) {
        std::cout << "Pose caricate: " << poses.size() << '\n'
                  << "Calibrazione caricata: " << calib_file << '\n';
    }
    else {
        std::cerr << "Ground truth non disponibile. "
                  << "L'ICP verrà eseguito senza calcolare gli errori.\n";
    }

    // --------------------------------------------------
    // 4. Apertura del CSV
    // --------------------------------------------------

    if (!csv_path.parent_path().empty()) {
        fs::create_directories(csv_path.parent_path());
    }

    std::ofstream csv(csv_path);

    if (!csv.is_open()) {
        std::cerr << "Impossibile creare il CSV: "
                  << csv_path << '\n';
        return 1;
    }

    csv << "sequence,src_idx,tgt_idx,n_src,n_tgt,"
        << "iters,dmax,t_grid,t_apply,t_nn,t_reduce,"
        << "t_solve,t_total,rot_err,trans_err\n";

    // --------------------------------------------------
    // 5. ICP su tutte le coppie consecutive della sequenza
    // --------------------------------------------------

    int successful_pairs = 0;
    int failed_pairs = 0;

    for (std::size_t i = 1; i < scans.size(); ++i) {
        const fs::path& src_file = scans[i];
        const fs::path& tgt_file = scans[i - 1];

        const int src_idx = std::stoi(src_file.stem().string());
        const int tgt_idx = std::stoi(tgt_file.stem().string());

        std::cout << "\n------------------------------------\n"
                  << "Coppia " << src_file.filename()
                  << " -> " << tgt_file.filename() << '\n';

        Rigid gt{};
        const Rigid* gt_ptr = nullptr;

        if (have_gt &&
            src_idx >= 0 && tgt_idx >= 0 &&
            src_idx < static_cast<int>(poses.size()) &&
            tgt_idx < static_cast<int>(poses.size())) {

            gt = compute_ground_truth(
                poses, Tr, src_idx, tgt_idx
            );

            gt_ptr = &gt;
        }

        // Se --out viene passato, salva ogni coppia con un nome diverso.
        std::string pair_out_path;

        if (!out_path.empty()) {
            fs::path output(out_path);

            if (output.extension() == ".ply") {
                pair_out_path =
                    (output.parent_path() /
                     (output.stem().string() + "_" +
                      std::to_string(tgt_idx) + "_" +
                      std::to_string(src_idx) + ".ply")).string();
            }
            else {
                fs::create_directories(output);
                pair_out_path =
                    (output /
                     (sequence + "_" +
                      std::to_string(tgt_idx) + "_" +
                      std::to_string(src_idx) + ".ply")).string();
            }
        }

        const int result = analyze_sequence(
            src_file.string(),
            tgt_file.string(),
            voxel,
            prm,
            gt_ptr,
            pair_out_path,
            csv,
            sequence,
            src_idx,
            tgt_idx
        );

        if (result == 0) {
            ++successful_pairs;
        }
        else {
            ++failed_pairs;
        }
    }

    csv.close();

    std::cout << "\n====================================\n"
              << "Elaborazione completata per la sequenza "
              << sequence << ".\n"
              << "Coppie riuscite: " << successful_pairs << '\n'
              << "Coppie fallite: " << failed_pairs << '\n'
              << "CSV salvato in: " << csv_path << '\n';

    return failed_pairs == 0 ? 0 : 1;
}