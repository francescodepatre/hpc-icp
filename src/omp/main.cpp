#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>
#include <cmath>
#include "icp.h"
#include "icp_omp.h"
#include "io.h"
#include "metrics.h"
#include "voxel.h"
#include "fusion.h"

namespace fs = std::filesystem;

using Mat4 = std::array<double, 16>;


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


static bool load_kitti_poses(
    const fs::path& path,
    std::vector<Mat4>& poses)
{
    std::ifstream f(path);

    if (!f.is_open()) {
        std::cerr << "Unable to open poses file: "
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
                std::cerr << "Invalid line in poses file: "
                          << path << '\n';
                return false;
            }
        }

        poses.push_back(to4x4(m));
    }

    return !poses.empty();
}


static bool load_kitti_calib(
    const fs::path& path,
    Mat4& Tr)
{
    std::ifstream f(path);

    if (!f.is_open()) {
        std::cerr << "Unable to open calibration file: "
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
                std::cerr << "Invalid Tr matrix: "
                          << path << '\n';
                return false;
            }
        }

        Tr = to4x4(m);

        return true;
    }

    std::cerr << "Tr matrix not found in: "
              << path << '\n';

    return false;
}


static Rigid compute_ground_truth(
    const std::vector<Mat4>& poses,
    const Mat4& Tr,
    int src_idx,
    int tgt_idx)
{
    if (src_idx < 0 ||
        tgt_idx < 0 ||
        static_cast<size_t>(src_idx) >= poses.size() ||
        static_cast<size_t>(tgt_idx) >= poses.size()) {
        throw std::out_of_range(
            "Invalid indices for ground truth"
        );
    }

    const Mat4 Tr_inv = inverse_rigid4(Tr);
    const Mat4 P_tgt_inv = inverse_rigid4(poses[tgt_idx]);

    Mat4 T = multiply4(Tr_inv, P_tgt_inv);
    T = multiply4(T, poses[src_idx]);
    T = multiply4(T, Tr);

    return to_rigid(T);
}


static bool is_bin_file(const fs::path& p)
{
    return p.extension() == ".bin";
}


static std::vector<fs::path> list_scans(const fs::path& sequence_dir)
{


    fs::path scan_dir = sequence_dir;

    bool has_bin = false;

    for (const auto& entry : fs::directory_iterator(scan_dir)) {
        if (entry.is_regular_file() && is_bin_file(entry.path())) {
            has_bin = true;
            break;
        }
    }

    if (!has_bin && fs::is_directory(sequence_dir / "velodyne")) {
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
        }
    );

    return scans;
}


static int scan_index(const fs::path& path)
{
    return std::stoi(path.stem().string());
}


static std::string detect_sequence_name(const fs::path& scans_root)
{

    if (scans_root.filename() == "velodyne") {
        return scans_root.parent_path().filename().string();
    }


    return scans_root.filename().string();
}

int main(int argc, char** argv)
{
    if (argc < 5) {
        std::cerr
            << "Usage:\n"
            << argv[0]
            << " <scans_dir> <results.csv>"
            << " <poses_dir> <calib_sequences_dir>"
            << " [--dmax D] [--iters K] [--voxel V]"
            << " [--threads N] [--out DIR]\n\n"
            << "Example:\n"
            << argv[0]
            << " ../data/data_odometry_velodyne/dataset/sequences"
            << " results/omp.csv"
            << " data/data_odometry_poses/dataset/poses"
            << " data/data_odometry_calib/dataset/sequences"
            << " --voxel 0.3 --dmax 1.5 --iters 30 --threads 8\n";

        return 1;
    }

    const fs::path scans_root = argv[1];
    const fs::path csv_path = argv[2];
    const fs::path poses_dir = argv[3];
    const fs::path calib_root = argv[4];

    IcpParams prm;
    float voxel = 0.0f;
    int threads = 0;


    fs::path out_dir;

    for (int i = 5; i < argc; ++i) {
        const std::string arg = argv[i];

        if (i + 1 >= argc) {
            std::cerr << "Argument without value: "
                      << arg << '\n';
            return 1;
        }

        const char* value = argv[++i];

        if (arg == "--dmax") {
            prm.dmax = std::strtof(value, nullptr);
        }
        else if (arg == "--iters") {
            prm.iters = std::atoi(value);
        }
        else if (arg == "--voxel") {
            voxel = std::strtof(value, nullptr);
        }
        else if (arg == "--threads") {
            threads = std::atoi(value);
        }
        else if (arg == "--out") {
            out_dir = value;
        }
        else {
            std::cerr << "Unknown option: "
                      << arg << '\n';
            return 1;
        }
    }

    if (!fs::is_directory(scans_root)) {
        std::cerr << "Invalid scans directory: "
                  << scans_root << '\n';
        return 1;
    }

    if (prm.iters <= 0 || prm.dmax <= 0.0f || voxel < 0.0f) {
        std::cerr << "Invalid parameters: iters and dmax must "
                     "be positive, voxel must be non-negative.\n";
        return 1;
    }

    if (threads > 0) {
        omp_set_num_threads(threads);
    }

    try {
        if (!csv_path.parent_path().empty()) {
            fs::create_directories(csv_path.parent_path());
        }

        if (!out_dir.empty()) {
            fs::create_directories(out_dir);
        }
    }
    catch (const fs::filesystem_error& e) {
        std::cerr << "Error creating directories: "
                  << e.what() << '\n';
        return 1;
    }

    std::ofstream csv(csv_path);

    if (!csv.is_open()) {
        std::cerr << "Unable to create CSV: "
                  << csv_path << '\n';
        return 1;
    }

    csv << "sequence,src_idx,tgt_idx,"
        << "n_src,n_tgt,iters,dmax,voxel,threads,"
        << "t_grid,t_apply,t_nn,t_reduce,t_solve,t_total,"
        << "rot_err,trans_err\n";

    std::cout << "OpenMP threads: "
              << omp_get_max_threads() << '\n';

    std::cout << "Scans directory: "
              << scans_root << '\n';

    std::cout << "CSV: " << csv_path << "\n\n";

    std::vector<fs::path> sequence_dirs;


bool contains_bin = false;

for (const auto& entry : fs::directory_iterator(scans_root)) {
    if (entry.is_regular_file() && entry.path().extension() == ".bin") {
        contains_bin = true;
        break;
    }
}

if (contains_bin) {
    sequence_dirs.push_back(scans_root);
}
else {

    for (const auto& entry : fs::directory_iterator(scans_root)) {
        if (entry.is_directory()) {
            sequence_dirs.push_back(entry.path());
        }
    }

    std::sort(
        sequence_dirs.begin(),
        sequence_dirs.end(),
        [](const fs::path& a, const fs::path& b) {
            return a.filename().string() < b.filename().string();
        }
    );
}

    size_t total_pairs = 0;
    size_t failed_pairs = 0;

    for (const fs::path& sequence_dir : sequence_dirs) {
        const std::string sequence = detect_sequence_name(sequence_dir);

        std::cout << "\n========================================\n";
        std::cout << "Sequence " << sequence << '\n';
        std::cout << "========================================\n";

        std::vector<fs::path> scans;

        try {
            scans = list_scans(sequence_dir);
        }
        catch (const fs::filesystem_error& e) {
            std::cerr << "Error while reading "
                      << sequence_dir << ": "
                      << e.what() << '\n';
            continue;
        }

        if (scans.size() < 2) {
            std::cerr << "Fewer than two scans in "
                      << sequence_dir << '\n';
            continue;
        }


PointCloud fused_map;
Rigid global_pose = rigid_identity();
bool fusion_tracking_valid = true;
bool fusion_initialized = false;


{
    PointCloud first_scan;
    std::string err;

    if (load_cloud(scans[0].string(), first_scan, &err) &&
        !first_scan.empty()) {

        if (voxel > 0.0f) {
            first_scan = voxel_downsample(first_scan, voxel);
        }

        if (!first_scan.empty()) {
            fused_map = std::move(first_scan);
            global_pose = rigid_identity();
            fusion_initialized = true;

            std::cout << "Map initialized with "
                      << fused_map.size()
                      << " points.\n";
        }
    }

    if (!fusion_initialized) {
        std::cerr << "Unable to initialize the map with "
                  << scans[0] << '\n';

        fusion_tracking_valid = false;
    }
}

        std::cout << "Scans found: "
                  << scans.size() << '\n';


        std::vector<Mat4> poses;
        Mat4 Tr = identity4();

        fs::path pose_path;
        fs::path calib_path;


        if (fs::is_regular_file(poses_dir)) {
            pose_path = poses_dir;
        } else {
            pose_path = poses_dir / (sequence + ".txt");
        }


        if (fs::is_regular_file(calib_root / "calib.txt")) {
            calib_path = calib_root / "calib.txt";
        } else {
            calib_path = calib_root / sequence / "calib.txt";
        }

        std::cout << "Poses file: " << pose_path << '\n';
        std::cout << "Calibration file: " << calib_path << '\n';

        const bool poses_ok = load_kitti_poses(pose_path, poses);
        const bool calib_ok = load_kitti_calib(calib_path, Tr);

        const bool gt_available = poses_ok && calib_ok;

        if (!gt_available) {
            std::cerr
                << "Warning: ground truth not available for "
                << sequence
                << ". ICP will run without GT errors.\n";
        }

        for (size_t i = 1; i < scans.size(); ++i) {
            const fs::path& src_path = scans[i];
            const fs::path& tgt_path = scans[i - 1];

            int src_idx;
            int tgt_idx;

            try {
                src_idx = scan_index(src_path);
                tgt_idx = scan_index(tgt_path);
            }
            catch (const std::exception& e) {
                std::cerr << "Invalid scan name: "
                          << e.what() << '\n';
                ++failed_pairs;
                continue;
            }

            std::cout << "\n[" << sequence << "] "
                      << src_path.filename().string()
                      << " -> "
                      << tgt_path.filename().string()
                      << '\n';

            PointCloud src;
            PointCloud tgt;
            std::string err;

            if (!load_cloud(src_path.string(), src, &err)) {
                std::cerr << "Error loading source: "
                          << err << '\n';
                ++failed_pairs;
                continue;
            }

            if (!load_cloud(tgt_path.string(), tgt, &err)) {
                std::cerr << "Error loading target: "
                          << err << '\n';
                ++failed_pairs;
                continue;
            }

            if (src.size() == 0 || tgt.size() == 0) {
                std::cerr << "Empty cloud, pair skipped.\n";
                ++failed_pairs;
                continue;
            }

            if (voxel > 0.0f) {
                src = voxel_downsample(src, voxel);
                tgt = voxel_downsample(tgt, voxel);
            }

            if (src.size() == 0 || tgt.size() == 0) {
                std::cerr << "Empty cloud after voxel "
                             "downsampling, pair skipped.\n";
                ++failed_pairs;
                continue;
            }

            Rigid gt{};
            bool pair_has_gt = false;

            if (gt_available &&
                src_idx >= 0 &&
                tgt_idx >= 0 &&
                static_cast<size_t>(src_idx) < poses.size() &&
                static_cast<size_t>(tgt_idx) < poses.size()) {

                gt = compute_ground_truth(
                    poses, Tr, src_idx, tgt_idx
                );

                pair_has_gt = true;
            }

            try {
                Timer total_timer;

                IcpResult r = icp_omp(src, tgt, prm);

                const double wall_total = total_timer.toc();

                if (r.rmse.empty() || r.matches.empty()) {
                    std::cerr
                        << "ICP produced no valid results.\n";

                    fusion_tracking_valid = false;
                    ++failed_pairs;
                    continue;
                }

                std::cout
                    << "Source points: " << src.size()
                    << ", target: " << tgt.size() << '\n';

                std::cout
                    << "Initial RMSE: " << r.rmse.front()
                    << ", final: " << r.rmse.back()
                    << ", valid correspondences: "
                    << r.matches.back() << '\n';

                std::cout << "Estimated T (source -> target):\n";

                for (int row = 0; row < 3; ++row) {
                    std::cout
                        << "  "
                        << std::fixed << std::setprecision(6)
                        << r.T.R[3 * row] << " "
                        << r.T.R[3 * row + 1] << " "
                        << r.T.R[3 * row + 2] << " "
                        << r.T.t[row] << '\n';
                }


bool transform_valid = true;

for (double value : r.T.R) {
    if (!std::isfinite(value)) {
        transform_valid = false;
        break;
    }
}

for (double value : r.T.t) {
    if (!std::isfinite(value)) {
        transform_valid = false;
        break;
    }
}

const bool icp_valid =
    transform_valid &&
    std::isfinite(r.rmse.back()) &&
    r.matches.back() >= 3;

if (fusion_tracking_valid &&
    fusion_initialized &&
    icp_valid) {


    global_pose = compose_transforms(global_pose, r.T);


    append_transformed(fused_map, src, global_pose);


    if (voxel > 0.0f) {
        fused_map = voxel_downsample(fused_map, voxel);
    }

    std::cout
        << "Fusion completed for scan "
        << src_idx
        << ". Points in map: "
        << fused_map.size()
        << '\n';
}
else {
    if (fusion_tracking_valid) {
        std::cerr
            << "Fusion stopped: invalid ICP transform "
            << "for "
            << src_idx << " -> " << tgt_idx
            << '\n';
    }

    fusion_tracking_valid = false;
}

                double rot_err = -1.0;
                double trans_err = -1.0;

                if (pair_has_gt) {
                    pose_error(r.T, gt, rot_err, trans_err);

                    std::cout
                        << "Rotation error: "
                        << rot_err << " degrees, "
                        << "translation error: "
                        << trans_err << " m\n";
                }
                else {
                    std::cout
                        << "Ground truth not available "
                        << "for this pair.\n";
                }

                const IcpTimings& tm = r.tm;

                std::cout
                    << "Timings (s): grid " << tm.grid
                    << " | apply " << tm.apply
                    << " | NN " << tm.nn
                    << " | reduce " << tm.reduce
                    << " | solve " << tm.solve
                    << " | ICP total " << tm.total
                    << " | measured total " << wall_total
                    << '\n';

                csv << sequence << ','
                    << src_idx << ','
                    << tgt_idx << ','
                    << src.size() << ','
                    << tgt.size() << ','
                    << prm.iters << ','
                    << prm.dmax << ','
                    << voxel << ','
                    << omp_get_max_threads() << ','
                    << tm.grid << ','
                    << tm.apply << ','
                    << tm.nn << ','
                    << tm.reduce << ','
                    << tm.solve << ','
                    << wall_total << ','
                    << rot_err << ','
                    << trans_err << '\n';

                csv.flush();


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
                            o
                        );

                        aligned.x[k] = static_cast<float>(o[0]);
                        aligned.y[k] = static_cast<float>(o[1]);
                        aligned.z[k] = static_cast<float>(o[2]);
                    }

                    std::ostringstream name;

                    name << sequence
                         << "_"
                         << std::setw(6) << std::setfill('0')
                         << src_idx
                         << "_to_"
                         << std::setw(6) << std::setfill('0')
                         << tgt_idx
                         << ".ply";

                    const fs::path ply_path = out_dir / name.str();

                    if (!save_ply(
                            ply_path.string(), aligned, &err)) {
                        std::cerr
                            << "Error saving PLY: "
                            << err << '\n';
                    }
                }

                ++total_pairs;
            }
            catch (const std::exception& e) {
                std::cerr
                    << "Error during ICP for "
                    << src_path << " -> " << tgt_path
                    << ": " << e.what() << '\n';

                ++failed_pairs;
            }
        }

if (fusion_initialized && !fused_map.empty()) {

    fs::path fusion_dir;

    if (!out_dir.empty()) {
        fusion_dir = out_dir;
    }
    else if (!csv_path.parent_path().empty()) {
        fusion_dir = csv_path.parent_path();
    }
    else {
        fusion_dir = ".";
    }

    try {
        fs::create_directories(fusion_dir);

        const fs::path fused_path =
            fusion_dir / (sequence + "_fused_omp.ply");

        std::string err;

        if (save_ply(fused_path.string(), fused_map, &err)) {
            std::cout
                << "\nFused map saved in: "
                << fused_path
                << "\nTotal points: "
                << fused_map.size()
                << '\n';
        }
        else {
            std::cerr
                << "Error saving fused map: "
                << err << '\n';
        }
    }
    catch (const fs::filesystem_error& e) {
        std::cerr
            << "Error in fused map directory: "
            << e.what() << '\n';
    }
}
    }

    csv.close();

    std::cout << "\n========================================\n";
    std::cout << "Benchmark terminated\n";
    std::cout << "Pairs completed: " << total_pairs << '\n';
    std::cout << "Pairs failed/skipped: " << failed_pairs << '\n';
    std::cout << "CSV saved in: " << csv_path << '\n';
    std::cout << "========================================\n";

    return failed_pairs > 0 ? 1 : 0;
}