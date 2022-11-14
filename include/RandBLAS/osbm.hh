#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#ifndef RandBLAS_OSBM_HH
#define RandBLAS_OSBM_HH

namespace RandBLAS::osbm {
    
template<typename T>
void OSBM(int64_t n_rows, int64_t n_cols, T *V, T *lev);

template<typename T>
void OSBMtest(int64_t n_rows, int64_t n_cols, T (*V), T (*lev));

template<typename T>
int check_levscores(int64_t n_rows, int64_t n_cols, T (*lev));

template<typename T>
T levscore_test(int64_t n_rows, int64_t n_cols, T (*V), T (*lev));

template<typename T>
T orthogonality_test(int64_t n_rows, int64_t n_cols, T (*V), int64_t ldc);

template<typename T>
int check_majorization(int64_t n_rows, int64_t n_cols, T (*rownorms), T (*lev));

template<typename T> 
T sgn(T val);

}
#endif
