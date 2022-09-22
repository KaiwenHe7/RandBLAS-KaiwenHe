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


template void larf<float>(char side, int64_t m, int64_t n, float *v, int64_t incv, float tau, float *C, int64_t ldc);
template void larf<double>(char side, int64_t m, int64_t n, double *v, int64_t incv, double tau, double *C, int64_t ldc);

template void print_colmaj<float>(uint64_t n_rows, uint64_t n_cols, float *a, char label[]);
template void print_colmaj<double>(uint64_t n_rows, uint64_t n_cols, double *a, char label[]);

template void genmat<float>(int64_t n_rows, int64_t n_cols, float* mat, uint64_t seed);
template void genmat<double>(int64_t n_rows, int64_t n_cols, double* mat, uint64_t seed);
} // end namespace RandBLAS::util
