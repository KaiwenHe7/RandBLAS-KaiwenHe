#include <blas.hh>
#include <lapack.hh>
#include <RandBLAS.hh>

#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <chrono>

template<typename T, size_t rows, size_t cols>
void print_mat(double M[rows][cols]);

template <typename T, size_t rows, size_t cols>
void printval(T (*array)[rows][cols]){
    int i,j;
    T rownorms[rows];
    T rowtemp[cols];
    /**array[0][0] = 1.0;
    std::cout << *array[0][1] << '\n';*/
    /*for (i=0; i<rows; i++) {
        for (j=0; j<cols; j++) {
            rowtemp[j] = (*array)[i][j];
            std::cout << (*array)[i][j] << '\n';
        }
        rownorms[i] = blas::dot(cols,rowtemp,1,rowtemp,1);
    }

    for (i=0;i<rows;i++) {
        std::cout << rownorms[i] << '\n';
    }*/
    /*double* critrow = (*array)[2];*/
   /* std::cout << array[0][1] << "\n";*/
}

template<typename T, size_t len>
void rot(T (*array)[len]){
    blas::rot(4, array[1], 1, array[2], 1, 0.1, 0.8);
}

template<typename T, size_t len>
double dotprod(T (*array)[len]){
    /*return blas::dot(3, &(*array)[0], 1, &(*array)[9],1);*/
    return blas::dot(3, &(*array)[0], 1, &(*array)[9],1);
}

template<typename T>
void test_passpartofarr(int64_t len, T *array){
    for (int i=0; i<len; i++){
        std::cout << array[i] << "\n";
    }
}

template <typename T>
void larfp(char side, int64_t m, int64_t n, T *v, int64_t incv, T tau, T *C, int64_t ldc) {
    if (side == 'R') {
        T w[m];
        blas::gemv(blas::Layout::RowMajor, blas::Op::NoTrans, m, n, 1, C, ldc, v, incv, 0, w, 1);
        blas::geru(blas::Layout::RowMajor, m, n, -1*tau, w, 1, v, incv, C, ldc);
    }
}

int main( int argc, char *argv[] ) {
    int i;

    double V[12] = {0,1,2,3,4,5,6,7,8,9,10,11};
    double w[3] = {1};
    larfp<double>('R', 4,1,w,1,2,&V[2],3);
    for (i=0; i<12; i++) {
        std::cout << V[i] << '\n';
    }
    

    
    return 0;
}

template<typename T, size_t rows, size_t cols>
void print_mat(double M[rows][cols]) {
    for (int i=0;i<rows;i++){
        std::cout << std::endl;
        std::cout << "|";
        for (int k=0; k<cols; k++){
            if(k>0) std::cout << "|";
            std::cout << M[i][k];
            if(k==cols-1) std::cout << "|";
        }
    }
}


