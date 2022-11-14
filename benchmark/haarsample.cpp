#include <blas.hh>
#include <lapack.hh>
#include <RandBLAS.hh>
#include <gtest/gtest.h>

#include <math.h>
#include <unistd.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <random>

template<typename T>
void apply_haar(int64_t n_rows, int64_t n_cols, T *mat, int64_t ldc, uint32_t seed);

template<typename T>
void gen_rmat_haar(int64_t n_rows, int64_t n_cols, T *mat, int32_t seed);

template<typename T>
void larf(char side, int64_t m, int64_t n, T *v, int64_t incv, T tau, T *C, int64_t ldc);

template<typename T>
void genlarf(int64_t len, T *x, int64_t ldx);

int main() {
    return 0;
}

template<typename T>
void apply_haar(int64_t n_rows, int64_t n_cols, T *mat, int64_t ldc, uint32_t seed) {
    int i,j;
    T signu0;              // Stores the sign of u[0] to avoid cancellation
    T u[n_cols];           // vector u to store random normal values. Used in the construction of Householder.
    T norm;                // Stores the norm of u to normalize u
    for (i=n_cols-1; i>0; i--) {
        RandBLAS::dense_op::gen_rmat_norm<T>(1, i+1, u, seed);
        signu0 = RandBLAS::osbm::sgn<T>(u[0]);
        u[0] = u[0] + signu0 * blas::nrm2(i+1,u,1);
        norm = blas::nrm2(i+1, u, 1);
        blas::scal(i+1, 1/norm, u, 1);
        RandBLAS::util::larf<T>('R', n_rows, i+1, &u[0], 1, 2, &mat[n_cols-i-1], ldc);
        blas::scal(n_rows, signu0, &mat[n_cols-i+1], ldc);
    }
}


template<typename T>
void gen_rmat_haar(int64_t n_rows, int64_t n_cols, T *mat, int32_t seed) {
    int i;
    for (i=0; i<n_rows*n_cols; i++) {
        mat[i] = 0;
    }
    for (i=0;i<n_cols; i++) {
        mat[i*n_cols + i] = 1;
    }
    apply_haar<T>(n_rows, n_cols, mat, n_cols, seed);
}

template<typename T>
void larf(char side, int64_t m, int64_t n, T *v, int64_t incv, T tau, T *C, int64_t ldc) {
    if (side == 'R') {
        T w[m];
        blas::gemv(blas::Layout::RowMajor, blas::Op::NoTrans, m, n, 1, C, ldc, v, incv, 0, w, 1);
        blas::geru(blas::Layout::RowMajor, m, n, -1*tau, w, 1, v, incv, C, ldc);
    }
    if (side == 'L') {
        T w[m];
        blas::gemv(blas::Layout::RowMajor, blas::Op::Trans, m, n, 1, C, ldc, v, 1, 0, w, 1);
        blas::geru(blas::Layout::RowMajor, m, n, -1*tau, v, 1, w, 1, C, ldc);
    
    }
}
