cmake_minimum_required(VERSION 3.18)
project(hpc_icp LANGUAGES CXX)

# ---------------------------------------------------------------- opzioni
option(ENABLE_OPENMP "Compila la versione OpenMP"        ON)
option(ENABLE_MPI    "Compila la versione MPI/ibrida"    ON)
option(ENABLE_CUDA   "Compila la versione CUDA (se nvcc)" ON)

# ---------------------------------------------------------------- standard e flag
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# -march=native: compila sulla macchina dove esegui (login node = compute node?)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
add_compile_options(-Wall -Wextra)

# ---------------------------------------------------------------- librerie
if(ENABLE_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
endif()

if(ENABLE_MPI)
  find_package(MPI COMPONENTS CXX)
endif()

set(HAVE_CUDA OFF)
if(ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    # Compute capability della GPU, es. 75 (Turing), 80 (A100), 86 (Ampere consumer).
    # Imposta con: cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..
    # La trovi con: nvidia-smi --query-gpu=compute_cap --format=csv
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(HAVE_CUDA ON)
  else()
    message(STATUS "nvcc non trovato: la versione CUDA viene saltata")
  endif()
endif()

# ---------------------------------------------------------------- codice comune
# loader PLY/KITTI, griglia, ICP, SVD 3x3, metriche
file(GLOB COMMON_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/common/*.cpp)
add_library(icp_common STATIC ${COMMON_SRC})
target_include_directories(icp_common PUBLIC ${CMAKE_SOURCE_DIR}/src/common)

# ---------------------------------------------------------------- eseguibili
# Ogni versione viene creata solo se il file main esiste, cosi' puoi
# configurare il progetto anche prima di aver scritto tutto.

# 1) Sequenziale
if(EXISTS ${CMAKE_SOURCE_DIR}/src/seq/main.cpp)
  add_executable(icp_seq src/seq/main.cpp)
  target_link_libraries(icp_seq PRIVATE icp_common)
endif()

# 2) OpenMP
if(OpenMP_CXX_FOUND AND EXISTS ${CMAKE_SOURCE_DIR}/src/omp/main.cpp)
  file(GLOB OMP_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/omp/*.cpp)
  add_executable(icp_omp ${OMP_SRC})
  target_link_libraries(icp_omp PRIVATE icp_common OpenMP::OpenMP_CXX)
endif()


# 3) MPI (e ibrido MPI+OpenMP se OpenMP e' disponibile)
if(MPI_CXX_FOUND AND EXISTS ${CMAKE_SOURCE_DIR}/src/mpi/main.cpp)
  add_executable(icp_mpi src/mpi/main.cpp)
  target_link_libraries(icp_mpi PRIVATE icp_common MPI::MPI_CXX)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(icp_mpi PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(icp_mpi PRIVATE USE_OPENMP)
  endif()
endif()

# 4) CUDA
if(HAVE_CUDA AND EXISTS ${CMAKE_SOURCE_DIR}/src/cuda/main.cu)
  file(GLOB CUDA_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/cuda/*.cu)
  add_executable(icp_cuda ${CUDA_SRC})
  target_link_libraries(icp_cuda PRIVATE icp_common)
  set_target_properties(icp_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  # per profilare con Nsight Compute aggiungi: target_compile_options(icp_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()

# 5) Strumenti e test (opzionali)
if(EXISTS ${CMAKE_SOURCE_DIR}/src/tools/make_pair.cpp)
  add_executable(make_pair src/tools/make_pair.cpp)
  target_link_libraries(make_pair PRIVATE icp_common)
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/tests/test_seq.cpp)
  enable_testing()
  add_executable(test_seq tests/test_seq.cpp)
  target_link_libraries(test_seq PRIVATE icp_common)
  add_test(NAME test_seq COMMAND test_seq)
endif()

if(OpenMP_CXX_FOUND AND EXISTS ${CMAKE_SOURCE_DIR}/tests/test_omp.cpp)
  add_executable(test_omp tests/test_omp.cpp src/omp/icp_omp.cpp)
  target_include_directories(test_omp PRIVATE ${CMAKE_SOURCE_DIR}/src/omp)
  target_link_libraries(test_omp PRIVATE icp_common OpenMP::OpenMP_CXX)
  add_test(NAME test_omp COMMAND test_omp)
endif()

# ---------------------------------------------------------------- riepilogo
message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenMP     : ${OpenMP_CXX_FOUND}")
message(STATUS "MPI        : ${MPI_CXX_FOUND}")
message(STATUS "CUDA       : ${HAVE_CUDA}")