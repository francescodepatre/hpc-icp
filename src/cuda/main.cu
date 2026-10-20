#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
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
#include "icp_cuda.h"
#include "io.h"
#include "metrics.h"
#include "voxel.h"

namespace fs = std::filesystem;

using Mat4 = std::array<double, 16>;

struct Sequence {
    std::string name;
    fs::path scans_dir;
};

static Mat4 identity4() {
    Mat4 m{};
    m[0] = m[5] = m[10] = m[15] = 1.0;
    return m;
}

static Mat4 multiply4(const Mat4& a, const Mat4& b) {
    Mat4 c{};

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                c[4 * i + j] +=
                    a[4 * i + k] * b[4 * k + j];
            }
        }
    }

    return c;
}

static Mat4 inverse_rigid4(const Mat4& m) {
    Mat4 inv = identity4();

    // Inversa della rotazione: R^T.
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            inv[4 * i + j] = m[4 * j + i];
        }
    }

    // Traslazione inversa: -R^T t.
    for (int i = 0; i < 3; ++i) {
        inv[4 * i + 3] = 0.0;

        for (int j = 0; j < 3; ++j) {
            inv[4 * i + 3] -=
                inv[4 * i + j] * m[4 * j + 3];
        }
    }

    return inv;
}

static Rigid to_rigid(const Mat4& m) {
    Rigid T{};

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            T.R[3 * i + j] = m[4 * i + j];
        }

        T.t[i] = m[4 * i + 3];
    }

    return T;
}

static bool load_kitti_poses(
    const fs::path& path,
    std::vector<Mat4>& poses
) {
    std::ifstream f(path);

    if (!f) {
        std::cerr << "Impossibile aprire le pose: "
                  << path << '\n';
        return false;
    }

    poses.clear();

    std::string line;

    while (std::getline(f, line)) {
        if (line.find_first_not_of(" \t\r\n") ==
            std::string::npos) {
            continue;
        }

        std::istringstream iss(line);
        double v[12];

        for (double& x : v) {
            if (!(iss >> x)) {
                std::cerr << "Riga delle pose non valida in "
                          << path << '\n';
                return false;
            }
        }

        Mat4 m = identity4();

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                m[4 * i + j] = v[4 * i + j];
            }
        }

        poses.push_back(m);
    }

    return !poses.empty();
}

static bool load_kitti_calib(
    const fs::path& path,
    Mat4& Tr
) {
    std::ifstream f(path);

    if (!f) {
        std::cerr << "Impossibile aprire la calibrazione: "
                  << path << '\n';
        return false;
    }

    std::string line;

    while (std::getline(f, line)) {
        std::istringstream iss(line);

        std::string key;
        iss >> key;

        if (key != "Tr:" && key != "Tr") {
            continue;
        }

        double v[12];

        for (double& x : v) {
            if (!(iss >> x)) {
                std::cerr << "Trasformazione Tr non valida in "
                          << path << '\n';
                return false;
            }
        }

        Tr = identity4();

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                Tr[4 * i + j] = v[4 * i + j];
            }
        }

        return true;
    }

    std::cerr << "Trasformazione Tr non trovata in "
              << path << '\n';

    return false;
}

// Pose KITTI: world <- camera.
// Tr: camera <- Velodyne.
// Il risultato è la trasformazione source -> target
// espressa nel sistema di coordinate Velodyne.
static Rigid compute_ground_truth(
    const std::vector<Mat4>& poses,
    const Mat4& Tr,
    size_t src_idx,
    size_t tgt_idx
) {
    const Mat4 Tr_inv = inverse_rigid4(Tr);
    const Mat4 P_tgt_inv = inverse_rigid4(poses[tgt_idx]);

    Mat4 T = multiply4(Tr_inv, P_tgt_inv);
    T = multiply4(T, poses[src_idx]);
    T = multiply4(T, Tr);

    return to_rigid(T);
}

static bool is_bin_file(const fs::path& p) {
    return fs::is_regular_file(p) && p.extension() == ".bin";
}

static bool contains_bin(const fs::path& dir) {
    if (!fs::is_directory(dir)) {
        return false;
    }

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (is_bin_file(entry.path())) {
            return true;
        }
    }

    return false;
}

static std::vector<fs::path> list_scans(const fs::path& dir) {
    std::vector<fs::path> scans;

    if (!fs::is_directory(dir)) {
        return scans;
    }

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (is_bin_file(entry.path())) {
            scans.push_back(entry.path());
        }
    }

    std::sort(scans.begin(), scans.end());

    return scans;
}

static std::string sequence_name(const fs::path& path) {
    if (path.filename() == "velodyne") {
        return path.parent_path().filename().string();
    }

    return path.filename().string();
}

static std::vector<Sequence> discover_sequences(
    const fs::path& root
) {
    std::vector<Sequence> sequences;

    // Caso 1: è stata passata direttamente la directory velodyne.
    if (contains_bin(root)) {
        sequences.push_back({
            sequence_name(root),
            root
        });

        return sequences;
    }

    // Caso 2: è stata passata la directory della sequenza, ad esempio 00.
    const fs::path velodyne = root / "velodyne";

    if (contains_bin(velodyne)) {
        sequences.push_back({
            root.filename().string(),
            velodyne
        });

        return sequences;
    }

    // Caso 3: è stata passata la root che contiene più sequenze.
    if (fs::is_directory(root)) {
        for (const auto& entry : fs::directory_iterator(root)) {
            if (!entry.is_directory()) {
                continue;
            }

            const fs::path scan_dir = entry.path() / "velodyne";

            if (contains_bin(scan_dir)) {
                sequences.push_back({
                    entry.path().filename().string(),
                    scan_dir
                });
            }
        }
    }

    std::sort(
        sequences.begin(),
        sequences.end(),
        [](const Sequence& a, const Sequence& b) {
            return a.name < b.name;
        }
    );

    return sequences;
}

static fs::path resolve_pose_path(
    const fs::path& arg,
    const std::string& sequence
) {
    if (fs::is_regular_file(arg)) {
        return arg;
    }

    // Supporta anche un argomento come poses/00 senza estensione.
    fs::path with_extension = arg;
    with_extension += ".txt";

    if (fs::is_regular_file(with_extension)) {
        return with_extension;
    }

    // Altrimenti l'argomento è la directory poses.
    return arg / (sequence + ".txt");
}

static fs::path resolve_calib_path(
    const fs::path& arg,
    const std::string& sequence
) {
    if (fs::is_regular_file(arg)) {
        return arg;
    }

    if (fs::is_regular_file(arg / "calib.txt")) {
        return arg / "calib.txt";
    }

    return arg / sequence / "calib.txt";
}

static size_t scan_index(const fs::path& p, size_t fallback) {
    try {
        return static_cast<size_t>(
            std::stoull(p.stem().string())
        );
    } catch (...) {
        return fallback;
    }
}

static void print_usage(const char* program) {
    std::cerr
        << "Uso:\n"
        << program
        << " <scans_dir> <results.csv> <poses_file_or_dir>"
        << " <calib_dir_or_sequence>"
        << " [--dmax D] [--iters K] [--voxel V]"
        << " [--device N] [--block B]"
        << " [--block-nn B] [--block-other B]"
        << " [--out DIR]\n\n"
        << "Esempio:\n"
        << program
        << " ../data/data_odometry_velodyne/dataset/sequences/00/velodyne/"
        << " ../results/bench_cuda.csv"
        << " ../data/data_odometry_poses/dataset/poses/00.txt"
        << " ../data/data_odometry_calib/dataset/sequences/00"
        << " --voxel 0.3 --dmax 1.5 --iters 30"
        << " --device 0 --block 128\n";
}

int main(int argc, char** argv) {
    if (argc < 5) {
        print_usage(argv[0]);
        return 1;
    }

    const fs::path scans_root = argv[1];
    const fs::path csv_path = argv[2];
    const fs::path poses_arg = argv[3];
    const fs::path calib_arg = argv[4];

    IcpParams prm;
    float voxel = 0.0f;

    CudaOptions opt;
    std::string out_dir;

    // Argomenti opzionali.
    for (int i = 5; i < argc; ++i) {
        const std::string a = argv[i];

        if (i + 1 >= argc) {
            std::cerr << "Argomento senza valore: " << a << '\n';
            return 1;
        }

        const char* value = argv[++i];

        try {
            if (a == "--dmax") {
                prm.dmax = std::stof(value);
            } else if (a == "--iters") {
                prm.iters = std::stoi(value);
            } else if (a == "--voxel") {
                voxel = std::stof(value);
            } else if (a == "--device") {
                opt.device = std::stoi(value);
            } else if (a == "--block") {
                opt.block_nn = std::stoi(value);
                opt.block_other = opt.block_nn;
            } else if (a == "--block-nn") {
                opt.block_nn = std::stoi(value);
            } else if (a == "--block-other") {
                opt.block_other = std::stoi(value);
            } else if (a == "--out") {
                out_dir = value;
            } else {
                std::cerr << "Opzione sconosciuta: " << a << '\n';
                return 1;
            }
        } catch (const std::exception&) {
            std::cerr << "Valore non valido per " << a
                      << ": " << value << '\n';
            return 1;
        }
    }

    if (prm.dmax <= 0.0f || prm.iters <= 0 || voxel < 0.0f) {
        std::cerr << "Parametri non validi: dmax e iters devono essere "
                     "positivi, voxel non negativo.\n";
        return 1;
    }

    try {
        const std::vector<Sequence> sequences =
            discover_sequences(scans_root);

        if (sequences.empty()) {
            throw std::runtime_error(
                "Nessuna scansione .bin trovata in " +
                scans_root.string()
            );
        }

        if (!csv_path.parent_path().empty()) {
            fs::create_directories(csv_path.parent_path());
        }

        if (!out_dir.empty()) {
            fs::create_directories(out_dir);
        }

        std::ofstream csv(csv_path);

        if (!csv) {
            throw std::runtime_error(
                "Impossibile creare il CSV: " + csv_path.string()
            );
        }

        csv << "sequence,src_idx,tgt_idx,n_src,n_tgt,"
               "iters,dmax,voxel,device,sm_count,"
               "t_grid,t_apply,t_nn,t_reduce,t_solve,t_total,"
               "t_h2d,t_d2h,rot_err,trans_err\n";

        csv << std::fixed << std::setprecision(8);

        size_t total_pairs = 0;
        size_t failed_pairs = 0;

        std::cout << "Sequenze trovate: " << sequences.size() << '\n';

        for (const Sequence& seq : sequences) {
            const std::vector<fs::path> scans =
                list_scans(seq.scans_dir);

            if (scans.size() < 2) {
                std::cerr << "[" << seq.name
                          << "] Meno di due scansioni, salto.\n";
                continue;
            }

            const fs::path pose_path =
                resolve_pose_path(poses_arg, seq.name);

            const fs::path calib_path =
                resolve_calib_path(calib_arg, seq.name);

            std::vector<Mat4> poses;
            Mat4 Tr;

            const bool poses_ok =
                load_kitti_poses(pose_path, poses);

            const bool calib_ok =
                load_kitti_calib(calib_path, Tr);

            const bool gt_available = poses_ok && calib_ok;

            if (!gt_available) {
                throw std::runtime_error(
                    "Impossibile caricare il ground truth per la sequenza " +
                    seq.name
                );
            }

            std::cout << "\nSequenza " << seq.name
                      << ": " << scans.size() << " scansioni\n"
                      << "  Pose: " << pose_path << '\n'
                      << "  Calibrazione: " << calib_path << '\n';

            for (size_t i = 1; i < scans.size(); ++i) {
                const size_t src_idx = scan_index(scans[i], i);
                const size_t tgt_idx = scan_index(scans[i - 1], i - 1);

                if (src_idx >= poses.size() || tgt_idx >= poses.size()) {
                    std::cerr << "[" << seq.name << "] Coppia "
                              << src_idx << " -> " << tgt_idx
                              << ": indice fuori dal file delle pose, salto.\n";
                    ++failed_pairs;
                    continue;
                }

                std::cout << "[" << seq.name << "] "
                          << src_idx << " -> " << tgt_idx << '\n';

                std::string err;
                PointCloud src, tgt;

                Timer tio;

                if (!load_cloud(scans[i].string(), src, &err)) {
                    std::cerr << "Errore lettura sorgente: "
                              << err << '\n';
                    ++failed_pairs;
                    continue;
                }

                if (!load_cloud(scans[i - 1].string(), tgt, &err)) {
                    std::cerr << "Errore lettura target: "
                              << err << '\n';
                    ++failed_pairs;
                    continue;
                }

                const double t_io = tio.toc();

                if (voxel > 0.0f) {
                    src = voxel_downsample(src, voxel);
                    tgt = voxel_downsample(tgt, voxel);
                }

                if (src.empty() || tgt.empty()) {
                    std::cerr << "Nuvola vuota dopo il preprocessing, salto.\n";
                    ++failed_pairs;
                    continue;
                }

                CudaInfo ci{};
                IcpResult r;

                try {
                    r = icp_cuda(
                        src,
                        tgt,
                        prm,
                        rigid_identity(),
                        opt,
                        &ci
                    );
                } catch (const std::exception& e) {
                    std::cerr << "Errore CUDA: " << e.what() << '\n';
                    ++failed_pairs;
                    continue;
                }

                double rot_err = -1.0;
                double trans_err = -1.0;

                const Rigid gt = compute_ground_truth(
                    poses,
                    Tr,
                    src_idx,
                    tgt_idx
                );

                pose_error(r.T, gt, rot_err, trans_err);

                const IcpTimings& tm = r.tm;

                const size_t actual_iters = r.rmse.size();

                const double last_rmse =
                    r.rmse.empty() ? -1.0 : r.rmse.back();

                const size_t last_matches =
                    r.matches.empty() ? 0 : r.matches.back();

                std::cout
                    << "  Punti: " << src.size()
                    << " / " << tgt.size() << '\n'
                    << "  Iterazioni completate: " << actual_iters
                    << ", RMSE finale: " << last_rmse
                    << ", match validi: " << last_matches << '\n'
                    << "  Errore rotazione: " << rot_err
                    << " gradi, traslazione: " << trans_err << '\n'
                    << "  Tempi (s): grid=" << tm.grid
                    << ", apply=" << tm.apply
                    << ", nn=" << tm.nn
                    << ", reduce=" << tm.reduce
                    << ", solve=" << tm.solve
                    << ", totale=" << tm.total << '\n'
                    << "  I/O: " << t_io << " s\n";

                csv
                    << seq.name << ','
                    << src_idx << ','
                    << tgt_idx << ','
                    << src.size() << ','
                    << tgt.size() << ','
                    << actual_iters << ','
                    << prm.dmax << ','
                    << voxel << ','
                    << ci.device_name << ','
                    << ci.sm_count << ','
                    << tm.grid << ','
                    << tm.apply << ','
                    << tm.nn << ','
                    << tm.reduce << ','
                    << tm.solve << ','
                    << tm.total << ','
                    << ci.t_h2d << ','
                    << ci.t_d2h << ','
                    << rot_err << ','
                    << trans_err << '\n';

                csv.flush();

                if (!out_dir.empty()) {
                    PointCloud aligned;
                    aligned.resize(src.size());

                    for (size_t j = 0; j < src.size(); ++j) {
                        double o[3];

                        rigid_apply(
                            r.T,
                            src.x[j],
                            src.y[j],
                            src.z[j],
                            o
                        );

                        aligned.x[j] = static_cast<float>(o[0]);
                        aligned.y[j] = static_cast<float>(o[1]);
                        aligned.z[j] = static_cast<float>(o[2]);
                    }

                    std::ostringstream filename;

                    filename << seq.name << '_'
                             << std::setw(6) << std::setfill('0')
                             << src_idx << "_to_"
                             << std::setw(6) << std::setfill('0')
                             << tgt_idx << ".ply";

                    const fs::path output =
                        fs::path(out_dir) / filename.str();

                    if (!save_ply(output.string(), aligned, &err)) {
                        std::cerr << "Errore salvataggio PLY: "
                                  << err << '\n';
                    }
                }

                ++total_pairs;
            }
        }

        std::cout << "\nElaborazione CUDA terminata.\n"
                  << "Coppie elaborate: " << total_pairs << '\n'
                  << "Coppie saltate o fallite: " << failed_pairs << '\n'
                  << "CSV: " << csv_path << '\n';

        return failed_pairs > 0 ? 1 : 0;

    } catch (const std::exception& e) {
        std::cerr << "Errore: " << e.what() << '\n';
        return 1;
    }
}