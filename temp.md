##sequenza 00

mpirun --oversubscribe -np 4 ./icp_mpi     ../data/data_odometry_velodyne/dataset/sequences/00/velodyne/     ../results/bench_mpi_00.csv     ../data/data_odometry_poses/dataset/poses/00.txt     ../data/data_odometry_calib/dataset/sequences/00 --voxel 0.3 --dmax 1.5 --iters 30 --threads 8 --partition block --collective reduce

./icp_seq ../data/data_odometry_velodyne/dataset/sequences/00/velodyne/ ../results/bench_seq.csv ../data/data_odometry_poses/dataset/poses/00.txt ../data/data_odometry_calib/dataset/sequences/00 --voxel 0.3 --dmax 1.5 --iters 30 --out ../results/seq_00_ply

./icp_omp ../data/data_odometry_velodyne/dataset/sequences/00 ../results/bench_omp_00.csv ../data/data_odometry_poses/dataset/poses ../data/data_odometry_calib/dataset/sequences --voxel 0.3 --dmax 1.5 --iters 30 --threads 8 --out ../results/omp_00_ply

./icp_cuda ../data/data_odometry_velodyne/dataset/sequences/00 ../results/bench_cuda_00.csv ../data/data_odometry_poses/dataset/poses ../data/data_odometry_calib/dataset/sequences --voxel 0.3 --dmax 1.5 --iters 30 --device 0 --block 128

##sequenza 02

mpirun -np 4 ./icp_mpi     ../data/data_odometry_velodyne/dataset/sequences/02/velodyne/     ../results/bench_mpi_02.csv     ../data/data_odometry_poses/dataset/poses/02.txt     ../data/data_odometry_calib/dataset/sequences/02 --voxel 0.3 --dmax 1.5 --iters 30 --threads 8 --partition block --collective reduce

./icp_seq ../data/data_odometry_velodyne/dataset/sequences/00/velodyne/ ../results/bench_seq_02.csv ../data/data_odometry_poses/dataset/poses/02.txt ../data/data_odometry_calib/dataset/sequences/02 --voxel 0.3 --dmax 1.5 --iters 30 --out ../results/seq_02_ply

./icp_omp ../data/data_odometry_velodyne/dataset/sequences/02 ../results/bench_omp_02.csv ../data/data_odometry_poses/dataset/poses ../data/data_odometry_calib/dataset/sequences --voxel 0.3 --dmax 1.5 --iters 30 --threads 8 --out ../results/omp_02_ply

./icp_cuda ../data/data_odometry_velodyne/dataset/sequences/00 ../results/bench_cuda_02.csv ../data/data_odometry_poses/dataset/poses ../data/data_odometry_calib/dataset/sequences --voxel 0.3 --dmax 1.5 --iters 30 --device 0 --block 128

##sequenza 19

mpirun -np 4 ./icp_mpi     ../data/data_odometry_velodyne/dataset/sequences/19/velodyne/     ../results/bench_mpi_19.csv     ../data/data_odometry_poses/dataset/poses/19.txt     ../data/data_odometry_calib/dataset/sequences/19 --voxel 0.3 --dmax 1.5 --iters 30 --threads 8 --partition block --collective reduce

./icp_seq ../data/data_odometry_velodyne/dataset/sequences/19/velodyne/ ../results/bench_seq_19.csv ../data/data_odometry_poses/dataset/poses/19.txt ../data/data_odometry_calib/dataset/sequences/19 --voxel 0.3 --dmax 1.5 --iters 30 --out ../results/seq_19_ply

./icp_omp ../data/data_odometry_velodyne/dataset/sequences/19 ../results/bench_omp_19.csv ../data/data_odometry_poses/dataset/poses ../data/data_odometry_calib/dataset/sequences --voxel 0.3 --dmax 1.5 --iters 30 --threads 8 --out ../results/omp_19_ply

./icp_cuda ../data/data_odometry_velodyne/dataset/sequences/19 ../results/bench_cuda_19.csv ../data/data_odometry_poses/dataset/poses ../data/data_odometry_calib/dataset/sequences --voxel 0.3 --dmax 1.5 --iters 30 --device 0 --block 128
