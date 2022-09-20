#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#ifndef RandBLAS_UTIL_HH
#define RandBLAS_UTIL_HH

namespace RandBLAS::util {

template <typename T>
void genmat(
	int64_t n_rows,
	int64_t n_cols,
	T* mat,
	uint64_t seed);


//template<typename uint64_t, typename TA>
//void print_colmaj(uint64_t n_rows, uint64_t n_cols, double *a, char label[]);

template <typename T>
void larf(char side, int64_t m, int64_t n, T *v, int64_t incv, T tau, T *C, int64_t ldc);

template <typename T>
void print_colmaj(uint64_t n_rows, uint64_t n_cols, T *a, char label[]);

} // end namespace RandBLAS::util

#endif  // define RandBLAS_UTIL_HH
