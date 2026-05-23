cmake_minimum_required(VERSION 3.20)

project(OtsuOpenMPCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(otsu_benchmark
    src/main.cpp
    src/otsu_cuda.cu
)

target_link_libraries(otsu_benchmark PRIVATE OpenMP::OpenMP_CXX)

set_target_properties(otsu_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)