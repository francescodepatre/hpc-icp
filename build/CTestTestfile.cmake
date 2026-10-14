# CMake generated Testfile for 
# Source directory: /home/francesco-de-patre/Desktop/progetto_HPC
# Build directory: /home/francesco-de-patre/Desktop/progetto_HPC/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_seq "/home/francesco-de-patre/Desktop/progetto_HPC/build/test_seq")
set_tests_properties(test_seq PROPERTIES  _BACKTRACE_TRIPLES "/home/francesco-de-patre/Desktop/progetto_HPC/CMakeLists.txt;104;add_test;/home/francesco-de-patre/Desktop/progetto_HPC/CMakeLists.txt;0;")
add_test(test_omp "/home/francesco-de-patre/Desktop/progetto_HPC/build/test_omp")
set_tests_properties(test_omp PROPERTIES  _BACKTRACE_TRIPLES "/home/francesco-de-patre/Desktop/progetto_HPC/CMakeLists.txt;111;add_test;/home/francesco-de-patre/Desktop/progetto_HPC/CMakeLists.txt;0;")
add_test(test_mpi_np2 "/usr/bin/mpiexec" "-n" "2" "/home/francesco-de-patre/Desktop/progetto_HPC/build/test_mpi")
set_tests_properties(test_mpi_np2 PROPERTIES  _BACKTRACE_TRIPLES "/home/francesco-de-patre/Desktop/progetto_HPC/CMakeLists.txt;124;add_test;/home/francesco-de-patre/Desktop/progetto_HPC/CMakeLists.txt;0;")
add_test(test_mpi_np3 "/usr/bin/mpiexec" "-n" "3" "/home/francesco-de-patre/Desktop/progetto_HPC/build/test_mpi")
set_tests_properties(test_mpi_np3 PROPERTIES  _BACKTRACE_TRIPLES "/home/francesco-de-patre/Desktop/progetto_HPC/CMakeLists.txt;124;add_test;/home/francesco-de-patre/Desktop/progetto_HPC/CMakeLists.txt;0;")
