cmake_minimum_required(VERSION 3.10)

project(benchmark)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

message(STATUS "Checking for RandBLAS ... ")
find_package(RandBLAS REQUIRED)
message(STATUS "Done checking for RandBLAS. ...")

message(STATUS "Looking for BLAS++ ... ")
find_package(blaspp REQUIRED)
message(STATUS "Done looking for BLAS++.")

message(STATUS "Looking for LAPACK++ ...")
find_package(lapackpp REQUIRED)
message(STATUS "Done looking for LAPACK++.")

message(STATUS "Looking for gtest ...")
find_package(gtest REQUIRED)
message(STATUS "Done looking for gtest")

set(
    test_cxx
    test.cpp
)
add_executable(
    test  ${test_cxx}
)
target_include_directories(
    test PUBLIC ${Random123_DIR}
)
target_link_libraries(
    test PUBLIC RandBLAS blaspp lapackpp 
)

set(
    OSBM_cxx
    OSBM.cpp
)
add_executable(
    OSBM  ${OSBM_cxx}
)
target_include_directories(
    OSBM PUBLIC ${Random123_DIR}
)
target_link_libraries(
    OSBM PUBLIC RandBLAS blaspp lapackpp 
)


#set(
#    bench_three_qr_algs_cxx
#    bench_three_qr_algs.cpp
#)
#add_executable(
#    bench_three_qr_algs ${bench_three_qr_algs_cxx}
#)
#target_include_directories(
#    bench_three_qr_algs PUBLIC ${Random123_DIR}
#)
#target_link_libraries(
#    bench_three_qr_algs PUBLIC RandBLAS blaspp lapackpp
#)
#
#
#set(
#    bench_blocksizes_cxx
#    bench_blocksizes.cpp
#)
#add_executable(
#    bench_blocksizes ${bench_blocksizes_cxx}
#)
#target_include_directories(
#    bench_blocksizes PUBLIC ${Random123_DIR}
#)
#target_link_libraries(
#    bench_blocksizes PUBLIC RandBLAS blaspp lapackpp
#)
