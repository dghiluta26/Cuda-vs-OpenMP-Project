cmake_minimum_required(VERSION 3.20)

project(OtsuOpenMP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(otsu_benchmark
    src/main.cpp
)

target_link_libraries(otsu_benchmark PRIVATE OpenMP::OpenMP_CXX)
