# Parallel ICP for LiDAR Point Clouds

This project implements pairwise point-cloud registration with the Iterative Closest Point (ICP) algorithm. It includes a sequential CPU version and parallel implementations using OpenMP, MPI, hybrid MPI+OpenMP, and CUDA. A uniform spatial grid limits nearest-neighbor searches to nearby target points. The programs can process KITTI Odometry sequences, record per-frame-pair timings and accuracy metrics in CSV files, and optionally write registered or fused point clouds.

## Requirements

The C++ programs require a C++17 compiler and CMake 3.18 or newer. OpenMP and MPI are detected by CMake when their development libraries and compiler wrappers are available. CUDA is optional and requires the NVIDIA CUDA Toolkit and a compatible NVIDIA GPU. The default CUDA architecture in `CMakeLists.txt` is 75; set `CMAKE_CUDA_ARCHITECTURES` to match the target GPU when necessary.

The analysis and visualization scripts require Python 3.12 or newer and the packages listed in `scripts/pyproject.toml` (Matplotlib, NumPy, Open3D, and Pandas). The project includes a `uv.lock` file; use `uv sync` from the `scripts` directory to create the Python environment with uv, or install the listed dependencies in your own environment.

## Build

From the project root, configure and build the versions supported by the current machine:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

CMake enables OpenMP, MPI, and CUDA by default when it finds the required toolchains. To explicitly build without CUDA, for example on a machine without an NVIDIA compiler, use:

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF
cmake --build build-cpu -j
```

To select a CUDA architecture, pass its compute capability during configuration, for example:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j
```

The build directory contains `icp_seq`, `icp_omp`, `icp_mpi`, and, when CUDA is available, `icp_cuda`. The `make_pair` utility and available test executables are also built when their dependencies are found.

## KITTI data

Download the KITTI Odometry Velodyne scans, calibration files, and ground-truth poses from the [KITTI Odometry benchmark page](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). The scan archive is large. Extract the archives under `data/` and preserve this layout:

```text
data/
  data_odometry_velodyne/dataset/sequences/00/velodyne/000000.bin
  data_odometry_calib/dataset/sequences/00/calib.txt
  data_odometry_poses/dataset/poses/00.txt
```

Repeat the sequence directories for the sequences you want to process. The programs expect KITTI Velodyne scans as binary float records and use the calibration and pose files to transform scans and calculate error metrics. The archive supplied with this project may include the poses archive, but the scan and calibration archives may need to be downloaded separately.

## Run a sequence

The sequential executable takes a scan directory, output CSV path, one sequence's pose file, and that sequence's calibration directory. For example, from the project root:

```bash
./build/icp_seq \
  data/data_odometry_velodyne/dataset/sequences/00/velodyne \
  results/seq_00.csv \
  data/data_odometry_poses/dataset/poses/00.txt \
  data/data_odometry_calib/dataset/sequences/00 \
  --voxel 0.3 --dmax 1.5 --iters 30 \
  --out results/seq_00
```

The OpenMP executable processes sequences found under the scans root. Pose and calibration roots must contain matching sequence files/directories:

```bash
./build/icp_omp \
  data/data_odometry_velodyne/dataset/sequences \
  results/omp.csv \
  data/data_odometry_poses/dataset/poses \
  data/data_odometry_calib/dataset/sequences \
  --voxel 0.3 --dmax 1.5 --iters 30 --threads 8 \
  --out results/omp
```

The MPI executable uses the same sequence-root layout. Launch it with the MPI launcher available on your system. The example below uses four ranks and two OpenMP threads per rank:

```bash
mpirun -np 4 ./build/icp_mpi \
  data/data_odometry_velodyne/dataset/sequences \
  results/mpi.csv \
  data/data_odometry_poses/dataset/poses \
  data/data_odometry_calib/dataset/sequences \
  --voxel 0.3 --dmax 1.5 --iters 30 --threads 2 \
  --partition block --collective reduce \
  --out results/mpi
```

Use the scheduler's MPI launcher (such as `srun`) instead of `mpirun` when required by the cluster. Allocate enough CPUs for the number of ranks multiplied by threads per rank. For MPI-only execution, set `--threads 1`. `--partition` accepts `block` or `cyclic`; `--collective` accepts `reduce` or `allreduce`.

CUDA processes one sequence using the GPU. It accepts either a sequence-specific pose file and calibration directory or the corresponding roots, as shown here:

```bash
./build/icp_cuda \
  data/data_odometry_velodyne/dataset/sequences/00 \
  results/cuda_00.csv \
  data/data_odometry_poses/dataset/poses \
  data/data_odometry_calib/dataset/sequences \
  --voxel 0.3 --dmax 1.5 --iters 30 --device 0 --block 128 \
  --out results/cuda_00
```

Run the CUDA executable on a node with a GPU and a compatible driver. `--block` sets the CUDA block size for the nearest-neighbor and other kernels. Use `--block-nn` and `--block-other` to set them separately.

The main parameters are `--voxel` (voxel-grid downsampling size in metres), `--dmax` (maximum correspondence distance in metres), and `--iters` (maximum ICP iterations). Use the same parameter values and input frame pairs when comparing implementations. The optional `--out` argument writes point-cloud outputs; omit it when only CSV measurements are needed.

## Benchmarking and analysis

`scripts/bench.py` runs repeated measurements on selected frame pairs for sequential, OpenMP, MPI, and hybrid configurations. It performs one warm-up by default and interleaves repetitions to reduce ordering bias. From the project root, a small example is:

```bash
python3 scripts/bench.py \
  --data data --sequences 00 --pairs 10 --step 1 \
  --exe seq=build/icp_seq omp=build/icp_omp mpi=build/icp_mpi hybrid=build/icp_mpi \
  --threads 2 4 --mpi-processes 1 2 4 \
  --voxel 0.3 --dmax 1.5 --iters 30 --reps 5 \
  --out results/bench.csv
```

The benchmark script's default MPI launcher is `mpirun`. On a Slurm cluster, select the appropriate launcher with `--mpiexec srun` and provide any site-specific launch options with `--mpi-flags`. For example, OpenMPI oversubscription flags can be passed as `--mpi-flags=--oversubscribe`. Follow local cluster policies for resource allocation and MPI launch commands. The benchmark script does not include CUDA in its configuration matrix; run `icp_cuda` directly to collect CUDA measurements.

Analyze one or more benchmark CSV files and save summary tables and plots:

```bash
python3 scripts/analyze_bench.py results/bench.csv --outdir results/analysis
```

The analysis reports speedup, parallel efficiency, Karp-Flatt effective serial fraction, and MPI overhead where applicable. By default, speedup uses `t_total`; change the metric with `--metric`, for example `--metric t_nn`. To create the plots directly from a benchmark CSV, use:

```bash
python3 scripts/plot_results.py results/bench.csv --outdir results/plots --metric t_total
```

`scripts/visualizza_icp.py`, `scripts/visualizza_sequenza.py`, and `scripts/stampa_ply.py` provide point-cloud visualization and output utilities. Run a script with `--help` where supported to see its arguments.

## Output files

Each ICP executable writes a CSV with a row per processed frame pair. The CSV contains the sequence and frame indices, source and target point counts, ICP parameters, timings for grid construction, point transformation, nearest-neighbor search, reduction, and solve, total time, and rotation and translation errors when ground truth is available. The `--out` option controls optional registered point-cloud output; without it, the program writes the sequence-level fused cloud alongside the CSV. Store generated results outside the source directories to keep input data unchanged.

For valid performance comparisons, use the same sequences, frame pairs, ICP parameters, and machine allocation for every implementation. Record the CPU/GPU model, compiler and library versions, rank and thread counts, and CUDA block sizes with the results. Treat timings from different machines or allocations as separate experiments.

## Repository layout

```text
src/common/   Shared point-cloud I/O, voxel filtering, grid, ICP, Kabsch, and metrics
src/seq/      Sequential implementation
src/omp/      OpenMP implementation
src/mpi/      MPI and hybrid MPI+OpenMP implementation
src/cuda/     CUDA implementation
src/tools/    Supporting command-line utilities
tests/        Correctness tests for the available implementations
scripts/      Benchmarking, analysis, plotting, and visualization tools
data/         KITTI data location
results/      Recommended location for generated CSV files and point clouds
```
