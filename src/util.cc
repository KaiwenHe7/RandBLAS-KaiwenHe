#include <iostream>
#include <memory>
#include <string>
#include <stdexcept>
#include <stdio.h>
#include <Random123/philox.h>
#include <Random123/uniform.hpp>
#include "util.hh"

namespace RandBLAS::util {

template <typename T>
void genmat(
	int64_t n_rows,
	int64_t n_cols,
	T* mat,
	uint64_t seed)
{
	typedef r123::Philox2x64 CBRNG;
	CBRNG::key_type key = {{seed}};
	CBRNG::ctr_type ctr = {{0,0}};
	CBRNG g;
	uint64_t prod = n_rows * n_cols;
	for (uint64_t i = 0; i < prod; ++i)
	{
		ctr[0] = i;
		CBRNG::ctr_type rand = g(ctr, key);
		mat[i] = r123::uneg11<T>(rand.v[0]);
	}
}

template <typename T>
void print_colmaj(uint64_t n_rows, uint64_t n_cols, T *a, char label[])
{
	uint64_t i, j;
    T val;
	std::cout << "\n" << label << std::endl;
    for (i = 0; i < n_rows; ++i) {
        std::cout << "\t";
        for (j = 0; j < n_cols - 1; ++j) {
            val = a[i + n_rows * j];
            if (val < 0) {
				//std::cout << string_format("  %2.4f,", val);
                printf("  %2.4f,", val);
            } else {
				//std::cout << string_format("   %2.4f", val);
				printf("   %2.4f,", val);
            }
        }
        // j = n_cols - 1
        val = a[i + n_rows * j];
        if (val < 0) {
   			//std::cout << string_format("  %2.4f,", val); 
			printf("  %2.4f,", val);
		} else {
            //std::cout << string_format("   %2.4f,", val);
			printf("   %2.4f,", val);
		}
        printf("\n");
    }
    printf("\n");
    return;
}

template <typename T>
void larf(char side, int64_t m, int64_t n, T *v, int64_t incv, T tau, T *C, int64_t ldc) {
    if (side == 'R') {
        T w[m];
        blas::gemv(blas::Layout::RowMajor, blas::Op::NoTrans, m, n, 1, C, ldc, v, incv, 0, w, 1);
        blas::geru(blas::Layout::RowMajor, m, n, -1*tau, w, 1, v, incv, C, ldc);
    }
    /*if (side == 'L') {
        T w[m];
        blas::gemv(blas::Layout::RowMajor, blas::Op::Trans, m, n, 1, C, ldc, v, 1, 0, w, 1);
        blas::geru(blas::Layout::RowMajor, m, n, -1*tau, v, 1, w, 1, C, ldc);
    }*/
}

/*template<typename T>
void random_sample(int64_t n_rows, int64_t n_cols, T (*V), uint32_t seed) {
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
        RandBLAS::util::larf<T>('R', n_rows, i+1, &u[0], 1, 2, &V[n_cols-i-1], n_cols); 
        blas::scal(n_rows, signu0, &V[n_cols-i+1], n_cols);
    }
}*/

template void larf<float>(char side, int64_t m, int64_t n, float *v, int64_t incv, float tau, float *C, int64_t ldc);
template void larf<double>(char side, int64_t m, int64_t n, double *v, int64_t incv, double tau, double *C, int64_t ldc);

//template void random_sample<float>(int64_t n_rows, int64_t n_cols, float *V, uint32_t seed);
//template void random_sample<double>(int64_t n_rows, int64_t n_cols, double *V, uint32_t seed);

template void print_colmaj<float>(uint64_t n_rows, uint64_t n_cols, float *a, char label[]);
template void print_colmaj<double>(uint64_t n_rows, uint64_t n_cols, double *a, char label[]);

template void genmat<float>(int64_t n_rows, int64_t n_cols, float* mat, uint64_t seed);
template void genmat<double>(int64_t n_rows, int64_t n_cols, double* mat, uint64_t seed);
} // end namespace RandBLAS::util
