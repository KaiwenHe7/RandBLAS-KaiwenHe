#include <blas.hh>
#include <lapack.hh>
#include <RandBLAS.hh>

#include <math.h>
#include <unistd.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <random>

template<typename T>
void gen_osbm(T *V, T *lev, void (*gen_lev)(T *lev, int64_t n_rows, int64_t n_cols), int64_t n_rows, int64_t n_cols);

template<typename T>
void gen_rvec_lev(T *lev, int64_t n_rows, int64_t n_cols);

template<typename T>
void gen_kspiked_lev(T *lev, int64_t n_rows, int64_t n_cols);

void gen_sjlts(RandBLAS::sjlts::SJLT *S, uint64_t n_rows, uint64_t n_cols, uint64_t vec_nnz);

void vary_nnz(double *err, int64_t m, int64_t n, int64_t d, int64_t len);

void print_vec(double *vec, int64_t len);

void vary_d(double *err, int64_t m, int64_t n, int64_t len);

double cond(double *A, int64_t n_rows, int64_t n_cols);

double subspace_distortion(double *SU, int64_t n_rows, int64_t n_cols);

int main() {
    
    int64_t d = 50;          // number of rows for the sketching matrix
    int64_t m = 201;         // number of rows for the test matrix
    int64_t n = 20;          // number of columns for the test matrix
    
    double err_nnz[10];
    vary_nnz(err_nnz, m, n, d, 10);
        
    /*double vary_d_err[14];
    vary_d(vary_d_err, m, n, 14);*/
    
    /*double A[m*n];
    double lev[m];
    gen_osbm<double>(A, lev, gen_kspiked_lev, m, n);
    std::cout << cond(A, m, n) << '\n';*/

    return 0;
}
template<typename T>
void gen_osbm(T *V, T *lev, void (*gen_lev)(T *lev, int64_t n_rows, int64_t n_cols), int64_t n_rows, int64_t n_cols) {
    int i;
    std::fill(V, V+n_rows*n_cols, 0);
    for (i=0; i<n_cols; i++) {
        V[(n_rows-n_cols)*n_cols + i + i*n_cols] = 1;
    }
    gen_lev(lev, n_rows, n_cols); 
    RandBLAS::osbm::OSBM<double>(n_rows, n_cols, V, lev);

}

void gen_sjlts(RandBLAS::sjlts::SJLT *S, uint64_t n_rows_sketched, uint64_t n_cols, uint64_t nnz){
    S->ori = RandBLAS::sjlts::ColumnWise;
    S->n_rows = n_rows_sketched;
    S->n_cols = n_cols;
    S->vec_nnz = nnz;
    S->rows = new uint64_t[nnz*n_cols];
    S->cols = new uint64_t[nnz*n_cols];
    S->vals = new double[nnz*n_cols];
    RandBLAS::sjlts::fill_colwise(*S, 5, 0);
    for (int i = 0; i<nnz*n_cols; i++) {
        S->vals[i] = S->vals[i] / sqrt(nnz);
    }
}

void vary_nnz(double *err, int64_t m, int64_t n, int64_t d, int64_t len) {
    double A[m*n];
    double lev[m];
    double SA[d*n];
    gen_osbm<double>(A, lev, gen_kspiked_lev, m, n);
    std::cout << "Matrix norm error from varying nnz per col" << '\n';
    for (uint64_t nnz = 1; nnz < len+1; nnz++) {
        RandBLAS::sjlts::SJLT *S = new RandBLAS::sjlts::SJLT;
        gen_sjlts(S, d, m, nnz);
        
        std::fill(SA, SA+d*n, 0);
        RandBLAS::sjlts::sketch_cscrow(*S, n, A, SA, 1);
        err[nnz-1] = subspace_distortion(SA, d, n);
        std::cout << "nnz = " << nnz << ":  " << err[nnz-1] << '\n'; 

        delete S;
    }
}

/*void vary_d(double *err, int64_t m, int64_t n, int64_t len){
    double A[m*n];
    double lev[m];
    gen_rmat_osbm<double>(A, lev, m, n);
    int ind = 0; 
    std::cout << "Matrix norm error from varying sketch dim" << '\n';
    for (uint64_t d = n+10; d < len*10+n+1; d += 10) {

        RandBLAS::sjlts::SJLT *S = new RandBLAS::sjlts::SJLT;
        double *SA = new double[d*n];
        std::fill(SA, SA+d*n, 0);

        gen_sjlts(S, d, m, 1);
        
        RandBLAS::sjlts::sketch_cscrow(*S, n, A, SA, 1);
        err[ind] = subspace_distortion(SA, d, n); 
        std::cout << "d = " << d << ":  " << err[ind] << '\n'; 
        ind += 1;

        delete[] SA;
        delete S;
    }

}*/

void print_vec(double *vec, int64_t len){
    for (int i = 0; i < len; i++){
        std::cout << vec[i] << '\n';
    }
}

double cond(double *A, int64_t n_rows, int64_t n_cols) {
    int i;
    double val;
    std::complex<double> newA[n_rows*n_cols];
    std::complex<double> *U = NULL;
    std::complex<double> *VT = NULL;
    double S[n_cols];
    for (i = 0; i < n_rows*n_cols; i++) {
        newA[i] = (std::complex<double>) A[i];
    }
    lapack::gesvd(lapack::Job::NoVec, lapack::Job::NoVec, n_cols, n_rows, newA, n_cols, S, U, 1, VT, 1);

    double maxval = RandBLAS::osbm::sgn<double>(S[0])*S[0];
    double minval = RandBLAS::osbm::sgn<double>(S[0])*S[0];
    for (i = 0; i < n_cols; i++) {
        val = RandBLAS::osbm::sgn<double>(S[i])*S[i];
        if ( val > maxval ) {
            maxval = val;
        } else if ( val < minval ) {
            minval = val;
        }
    }
    return maxval/minval;
}

double subspace_distortion(double *SU, int64_t n_rows, int64_t n_cols) {
    double condSA = cond(SU, n_rows, n_cols);
    return (condSA - 1) / (condSA + 1);
}

template<typename T>
void gen_rvec_lev(T *lev, int64_t n_rows, int64_t n_cols) {
    int i;
    T sum = 0;
    RandBLAS::dense_op::gen_rmat_unif<double>(1, n_rows, lev, 0);
    blas::scal(n_rows, 0.5, lev, 1); 
    for (i=0; i<n_rows; i++) {
        lev[i] += 0.5;
        sum += lev[i];
    }
    blas::scal(n_rows, n_cols/sum, lev, 1);
    std::sort(lev, lev+n_rows);
}

template<typename T>
void gen_kspiked_lev(T *lev, int64_t n_rows, int64_t n_cols) {
    int i;
    int k = 5;
    gen_rvec_lev(&(lev[2*k]), n_rows-2*k, n_cols-k);
    for (i=0; i < k; i++) {
        lev[i] = 0.99999;
    }
    for (i=k; i < 2*k; i++) {
        lev[i] = 0.00001;
    }
    std::sort(lev, lev+n_rows);
}

