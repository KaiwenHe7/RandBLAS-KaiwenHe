cmake_minimum_required(VERSION 3.10)

project(benchmark)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

message(STATUS "Checking for hqrrp ... ")
find_package(RandBLAS REQUIRED)
message(STATUS "Done checking for hqrrp. ...")

message(STATUS "Looking for BLAS++ ... ")
find_package(blaspp REQUIRED)
message(STATUS "Done looking for BLAS++.")

message(STATUS "Looking for LAPACK++ ...")
find_package(lapackpp REQUIRED)
message(STATUS "Done looking for LAPACK++.")


set(
    bench_three_qr_algs_cxx
    bench_three_qr_algs.cpp
)
add_executable(
    bench_three_qr_algs ${bench_three_qr_algs_cxx}
)
target_include_directories(
    bench_three_qr_algs PUBLIC ${Random123_DIR}
)
target_link_libraries(
     bench_three_qr_algs PUBLIC hqrrp blaspp lapackpp
)


set(
    bench_blocksizes_cxx
    bench_blocksizes.cpp
)
add_executable(
    bench_blocksizes ${bench_blocksizes_cxx}
)
target_include_directories(
    bench_blocksizes PUBLIC ${Random123_DIR}
)
target_link_libraries(
    bench_blocksizes PUBLIC hqrrp blaspp lapackpp
)
