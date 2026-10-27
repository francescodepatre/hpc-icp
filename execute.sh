#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --time=04:00:00

module load cuda/12.4.1
module load cmake/3.29.4
module load gcc/13.2.0
module load gn8/8.3.0
module load openmpi4/4.1.8

#seq 00

srun --ntasks=4 --cpus-per-task=2 ./icp_mpi ../data/data_odometry_velodyne/dataset/sequences/00/velodyne/ ../results/result_00/bench_mpi_00.csv ../data/data_odometry_poses/dataset/poses/00.txt ../data/data_odometry_calib/dataset/sequences/00 --voxel 0.3 --dmax 1.5 --iters 30 --threads 2 --partition block --collective reduce

./icp_seq ../data/data_odometry_velodyne/dataset/sequences/00/velodyne/ ../results/result_00/bench_seq_00.csv ../data/data_odometry_poses/dataset/poses/00.txt ../data/data_odometry_calib/dataset/sequences/00 --voxel 0.3 --dmax 1.5 --iters 30 --out ../results/result_00/seq_00_ply

./icp_omp ../data/data_odometry_velodyne/dataset/sequences/00 ../results/result_00/bench_omp_00.csv ../data/data_odometry_poses/dataset/poses ../data/data_odometry_calib/dataset/sequences --voxel 0.3 --dmax 1.5 --iters 30 --threads 8 --out ../results/result_00/omp_00_ply




srun --ntasks=4 --cpus-per-task=2 ./icp_mpi ../data/data_odometry_velodyne/dataset/sequences/00/velodyne/ ../results/result_00/bench_mpi_00.csv ../data/data_odometry_poses/dataset/poses/00.txt ../data/data_odometry_calib/dataset/sequences/00 --voxel 0.3 --dmax 1.5 --iters 30 --threads 2 --partition block --collective reduce


#./icp_cuda ../data/data_odometry_velodyne/dataset/sequences/00 ../results/bench_cuda_00.csv ../data/data_odometry_poses/dataset/poses ../data/data_odometry_calib/dataset/sequences --voxel 0.3 --dmax 1.5 --iters 30 --device 0 --block 128


#mpirun -np 4 ./icp_mpi     ../data/data_odometry_velodyne/dataset/sequences/00/velodyne/     ../results/bench_mpi_00.csv     ../data/data_odometry_poses/dataset/poses/00.txt     ../data/data_odometry_calib/dataset/sequences/00 --voxel 0.3 --dmax 1.5 --iters 30 --threads 8 --partition block --collective reduce


srun --nodes=1 --ntasks=1 --cpus-per-task=8 --partition=cpu --qos=cpu --time=01:00:00 --pty bash

srun --ntasks=4 --cpus-per-task=2 ./icp_mpi ../data/data_odometry_velodyne/dataset/sequences/00/velodyne/ ../results/result_00/bench_mpi_00.csv ../data/data_odometry_poses/dataset/poses/00.txt ../data/data_odometry_calib/dataset/sequences/00 --voxel 0.3 --dmax 1.5 --iters 30 --threads 2 --partition block --collective reduce


mpirun -np 4 ./icp_mpi ../data/data_odometry_velodyne/dataset/sequences/00/velodyne/ ../results/result_00/bench_mpi_00.csv ../data/data_odometry_poses/dataset/poses/00.txt ../data/data_odometry_calib/dataset/sequences/00 --voxel 0.3 --dmax 1.5 --iters 30 --threads 2 --partition block --collective reduce

salloc --nodes=1 --ntasks=4 --cpus-per-task=2 --partition=cpu --qos=cpu --time=04:00:00


srun --ntasks=1 --cpus-per-task=4 --gres=gpu:1 


./icp_cuda  ../data/data_odometry_velodyne/dataset/sequences/00 ../results/result_00/cuda/bench_cuda_00.csv ../data/data_odometry_poses/dataset/poses ../data/data_odometry_calib/dataset/sequences --voxel 0.3 --dmax 1.5 --iters 30 --device 0 --block 128


srun --partition=gpu --qos=gpu --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=02:00:00 --pty bash -i