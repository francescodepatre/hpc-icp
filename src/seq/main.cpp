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
#include <utility>
#include <filesystem>
#include <algorithm>
#include <iomanip>

#include <array>
#include <sstream>
#include <vector>

#include "fusion.h"
#include <cmath>

using Mat4 = std::array<double, 16>;

static Mat4 identity4(){
    Mat4 M{};
    M[0]  = 1.0;
    M[5]  = 1.0;
    M[10] = 1.0;
    M[15] = 1.0;

    return M;
}

static Mat4 to4x4(const double m[12]){
    Mat4 M = identity4();

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            M[4 * i + j] = m[4 * i + j];
        }
    }
    return M;
}

static bool load_kitti_poses(const std::string& path, std::vector<Mat4>& poses){
    std::ifstream f(path);

    if (!f.is_open()) {
        std::cerr << "Cannot open pose file: " << path << '\n';
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
                std::cerr << "Invalid row in pose file: " << path << '\n';
                return false;
            }
        }
        poses.push_back(to4x4(m));
    }

    return !poses.empty();
}

static bool load_kitti_calib(const std::string& path, Mat4& Tr){
    std::ifstream f(path);

    if (!f.is_open()) {
        std::cerr << "Cannot open calibration file: " << path << '\n';
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
                std::cerr << "Invalid Tr matrix in: " << path << '\n';
                return false;
            }
        }

        Tr = to4x4(m);

        return true;
    }

    std::cerr << "Tr matrix not found in: " << path << '\n';

    return false;
}

static Mat4 multiply4(const Mat4& A, const Mat4& B){
    Mat4 C{};

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                C[4 * i + j] += A[4 * i + k] * B[4 * k + j];
            }
        }
    }

    return C;
}

static Mat4 inverse_rigid4(const Mat4& T){
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


static Rigid to_rigid(const Mat4& M){
    Rigid T{};

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            T.R[3 * i + j] = M[4 * i + j];
        }

        T.t[i] = M[4 * i + 3];
    }

    return T;
}

static Rigid compute_ground_truth( const std::vector<Mat4>& poses, const Mat4& Tr, int src_idx, int tgt_idx){
    if (src_idx < 0 || tgt_idx < 0 || src_idx >= static_cast<int>(poses.size()) || tgt_idx >= static_cast<int>(poses.size())) {
        throw std::out_of_range("Poses index out of bounds");
    }

    const Mat4& P_src = poses[src_idx];
    const Mat4& P_tgt = poses[tgt_idx];

    const Mat4 Tr_inv = inverse_rigid4(Tr);
    const Mat4 P_tgt_inv = inverse_rigid4(P_tgt);

    Mat4 T = multiply4(multiply4(multiply4(Tr_inv, P_tgt_inv),P_src),Tr);

    return to_rigid(T);
}

int analyze_sequence( const std::string& src_path, const std::string& tgt_path, float voxel, const IcpParams& prm, const Rigid* gt, const std::string& out_path, std::ofstream& csv, const std::string& sequence, int src_idx, int tgt_idx, Rigid* estimated_transform){
    std::string err;
    PointCloud src, tgt;
    Timer tio;
    if (!load_cloud(src_path, src, &err) || !load_cloud(tgt_path, tgt, &err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    const double t_io = tio.toc();
    std::printf("loaded: source %zu points, target %zu points (%.3f s)\n", src.size(), tgt.size(), t_io);

    if (voxel > 0.f) {
        Timer tv;
        src = voxel_downsample(src, voxel);
        tgt = voxel_downsample(tgt, voxel);
        std::printf("voxel %.4f: source %zu, target %zu (%.3f s)\n", voxel, src.size(), tgt.size(), tv.toc());
    }

    IcpResult r = icp_seq(src, tgt, prm);

    if (r.rmse.empty() || r.matches.empty()) {
    std::fprintf(stderr, "No valid results from ICP.\n");
    return 1;
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

if (!transform_valid || !std::isfinite(r.rmse.back()) || r.matches.back() < 3) {
    std::fprintf(stderr, "Invalid ICP transform.\n");
    return 1;
}

if (estimated_transform != nullptr) {
    *estimated_transform = r.T;
}

    std::printf("\nICP: dmax=%.4f iters=%d\n", prm.dmax, prm.iters);
    std::printf("RMSE first iter: %.6f  last: %.6f  (last valid pair: %zu)\n", r.rmse.front(), r.rmse.back(), r.matches.back());
    std::printf("T estimated (source -> target):\n");
    for (int i = 0; i < 3; ++i){
        std::printf("  % .6f % .6f % .6f % .6f\n", r.T.R[3 * i], r.T.R[3 * i + 1], r.T.R[3 * i + 2], r.T.t[i]);
    }

    double rot = -1.0;
    double tr = -1.0;

    if (gt != nullptr) {
        pose_error(r.T, *gt, rot, tr);

        std::printf("Rotation error: %.5f degrees, ""traslation error: %.6f\n",rot, tr);
    } else {
        std::printf("Ground truth not available: errors not estimated.");
    }

    const IcpTimings& tm = r.tm;
    std::printf("\ntimes (s): grid %.4f | apply %.4f | nn %.4f | reduce %.4f | solve %.4f | total %.4f\n", tm.grid, tm.apply, tm.nn, tm.reduce, tm.solve, tm.total);

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
        std::printf("source allineated saved in %s\n", out_path.c_str());
    }
    return 0;
}

namespace fs = std::filesystem;

int main(int argc, char** argv){

    if (argc < 5) {
        std::fprintf( stderr, "Usage: %s <scans_dir> <result.csv> " "<pose_file> <calib_dir> " "[--dmax D] [--iters K] [--voxel V] [--out FILE]\n", argv[0]);
        return 1;
    }

    const fs::path scan_dir = argv[1];
    const fs::path csv_path = argv[2];
    const fs::path pose_file = argv[3];
    const fs::path calib_dir = argv[4];

    IcpParams prm;
    float voxel = 0.0f;
    std::string out_path;

    for (int i = 5; i < argc; ++i) {
        const std::string a = argv[i];

        if (i + 1 >= argc) {
            std::cerr << "Arg without value: " << a << '\n';
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
            std::cerr << "Unknown option: " << a << '\n';
            return 1;
        }
    }

    if (!fs::is_directory(scan_dir)) {
        std::cerr << "Invalid scans directory: " << scan_dir << '\n';
        return 1;
    }

    if (!fs::is_regular_file(pose_file)) {
        std::cerr << "Pose file not found: " << pose_file << '\n';
        return 1;
    }

    if (!fs::is_directory(calib_dir)) {
        std::cerr << "Calibration directory not found: " << calib_dir << '\n';
        return 1;
    }

    const std::string sequence = scan_dir.filename().string();

    std::cout << "\n====================================\n";
    std::cout << "Sequence selected: " << sequence << '\n';
    std::cout << "Directory scans: " << scan_dir << '\n';
    std::cout << "File pose: " << pose_file << '\n';
    std::cout << "Directory calibration: " << calib_dir << '\n';
    std::cout << "====================================\n";

    std::vector<fs::path> scans;

    for (const auto& entry : fs::directory_iterator(scan_dir)) {
        if (entry.is_regular_file() &&
            entry.path().extension() == ".bin") {
            scans.push_back(entry.path());
        }
    }

    std::sort(scans.begin(), scans.end(), [](const fs::path& a, const fs::path& b) {
        return std::stoi(a.stem().string()) < std::stoi(b.stem().string());
    });

    if (scans.size() < 2) {
        std::cerr << "Less than 2 files .bin found in " << scan_dir << '\n';
        return 1;
    }

    std::cout << "Scans found: " << scans.size() << '\n';

PointCloud fused_map;
Rigid global_pose = rigid_identity();

bool fusion_initialized = false;
bool fusion_tracking_valid = true;

{
    PointCloud first_scan;
    std::string err;

    if (load_cloud(scans[0].string(), first_scan, &err) && !first_scan.empty()) {
        if (voxel > 0.0f) {
            first_scan = voxel_downsample(first_scan, voxel);
        }

        if (!first_scan.empty()) {
            fused_map = std::move(first_scan);
            fusion_initialized = true;

            std::cout << "Initial map created with " << fused_map.size() << " points.\n";
        }
    }

    if (!fusion_initialized) {
        std::cerr << "Error: cannot load first scan.\n";
        return 1;
    }
}

    const fs::path calib_file = calib_dir / "calib.txt";

    std::vector<Mat4> poses;
    Mat4 Tr{};

    const bool poses_ok = load_kitti_poses(pose_file.string(), poses);

    const bool calib_ok = load_kitti_calib(calib_file.string(), Tr);

    const bool have_gt = poses_ok && calib_ok;

    if (have_gt) {
        std::cout << "Loaded poses: " << poses.size() << '\n' << "Loaded calibration: " << calib_file << '\n';
    }
    else {
        std::cerr << "Ground truth not available. " << '\n' << "ICP will be executed without error estimation.\n";
    }

    if (!csv_path.parent_path().empty()) {
        fs::create_directories(csv_path.parent_path());
    }

    std::ofstream csv(csv_path);

    if (!csv.is_open()) {
        std::cerr << "Cannot create il CSV file: " << csv_path << '\n';
        return 1;
    }

    csv << "sequence,src_idx,tgt_idx,n_src,n_tgt," << "iters,dmax,t_grid,t_apply,t_nn,t_reduce," << "t_solve,t_total,rot_err,trans_err\n";

    int successful_pairs = 0;
    int failed_pairs = 0;

    for (std::size_t i = 1; i < scans.size(); ++i) {
        const fs::path& src_file = scans[i];
        const fs::path& tgt_file = scans[i - 1];

        const int src_idx = std::stoi(src_file.stem().string());
        const int tgt_idx = std::stoi(tgt_file.stem().string());

        std::cout << "\n------------------------------------\n" << "Coppia " << src_file.filename() << " -> " << tgt_file.filename() << '\n';

        Rigid gt{};
        const Rigid* gt_ptr = nullptr;

        if (have_gt && src_idx >= 0 && tgt_idx >= 0 && src_idx < static_cast<int>(poses.size()) && tgt_idx < static_cast<int>(poses.size())) {

            gt = compute_ground_truth(poses, Tr, src_idx, tgt_idx);

            gt_ptr = &gt;
        }

        std::string pair_out_path;

        if (!out_path.empty()) {
            fs::path output(out_path);

            if (output.extension() == ".ply") {
                pair_out_path = (output.parent_path() / (output.stem().string() + "_" + std::to_string(tgt_idx) + "_" + std::to_string(src_idx) + ".ply")).string();
            }
            else {
                fs::create_directories(output);
                pair_out_path = (output / (sequence + "_" + std::to_string(tgt_idx) + "_" + std::to_string(src_idx) + ".ply")).string();
            }
        }

        Rigid estimated_transform{};

const int result = analyze_sequence( src_file.string(), tgt_file.string(), voxel, prm, gt_ptr, pair_out_path, csv, sequence, src_idx, tgt_idx, &estimated_transform);

if (result == 0) {
    ++successful_pairs;

    if (fusion_tracking_valid && fusion_initialized) {

        global_pose = compose_transforms(global_pose,estimated_transform);

        PointCloud src;
        std::string err;

        if (load_cloud(src_file.string(), src, &err)) {

            if (voxel > 0.0f) {
                src = voxel_downsample(src, voxel);
            }

            append_transformed(fused_map,src,global_pose);

            if (voxel > 0.0f) {
                fused_map = voxel_downsample(fused_map, voxel);
            }

            std::cout << "Fusion completed. Points in map: " << fused_map.size() << '\n';
        }
        else {
            std::cerr << "Error during scan loading" << "for fusion: " << err << '\n';
            fusion_tracking_valid = false;
        }
    }
}
else {
    ++failed_pairs;
    fusion_tracking_valid = false;
}
        if (result == 0) {
            ++successful_pairs;
        }
        else {
            ++failed_pairs;
        }
    }

if (fusion_initialized && !fused_map.empty()) {

    fs::path fused_path;

    if (!out_path.empty()) {
        fs::path output(out_path);

        if (output.extension() == ".ply") {
            fused_path = output.parent_path() / (output.stem().string() + "_fused.ply");
        }
        else {
            fused_path = output / (sequence + "_fused_seq.ply");
        }
    }
    else if (!csv_path.parent_path().empty()) {
        fused_path = csv_path.parent_path() / (sequence + "_fused_seq.ply");
    }
    else {
        fused_path = sequence + "_fused_seq.ply";
    }
    if (!fused_path.parent_path().empty()) {
        fs::create_directories(fused_path.parent_path());
    }

    std::string err;

    if (save_ply(fused_path.string(), fused_map, &err)) {
        std::cout << "\nFused map saved in: " << fused_path << '\n' << "Total points: " << fused_map.size() << '\n';

        if (!fusion_tracking_valid) {
            std::cout << "Attenzion: partial map, " << "recording chain has been interrupted.\n";
        }
    }
    else {
        std::cerr << "Error saving map: " << err << '\n';
    }
}

    csv.close();

    std::cout << "\n====================================\n" << "Sequence elaboration completed " << sequence << ".\n" << "Pairs completed: " << successful_pairs << '\n' << "Failed pairs: " << failed_pairs << '\n' << "CSV saved in: " << csv_path << '\n';
    return failed_pairs == 0 ? 0 : 1;
}