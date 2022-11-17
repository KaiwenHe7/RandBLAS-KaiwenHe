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
void OSBMmixed(int64_t n_rows, int64_t n_cols, T (*V), T (*lev)); 

template<typename T>
void OSBM(int64_t n_rows, int64_t n_cols, T (*V), T (*lev));

template<typename T>
T levscore_test(int64_t n_rows, int64_t n_cols, T (*V), T (*lev));

template<typename T>
T orthogonality_test(int64_t n_rows, int64_t n_cols, T (*V), int64_t ldc);

template<typename T>
int check_levscores(int64_t n_rows, int64_t n_cols, T (*lev));

template<typename T>
int check_majorization(int64_t n_rows, int64_t n_cols, T *rownorms, T *lev);

template <typename T> 
T sgn(T val);

int main(){
    int i;

    int64_t m = 100000;
    int64_t n = 1000;

    double *dV = new double[m*n];
    double *dlev = new double[m];

    std::fill(dV, dV+n*m, 0);
    for (i=0; i<n; i++) {
        dV[(m-n)*n + i + i*n] = 1;
    }

    double sum = 0;
    RandBLAS::dense_op::gen_rmat_unif<double>(1, m, dlev, 0);
    blas::scal(m, 0.5, dlev, 1); 
    for (i=0; i<m; i++) {
        dlev[i] += 0.5;
        sum += dlev[i];
    }
    blas::scal(m, n/sum, dlev, 1);
    std::sort(dlev, dlev+m);


    OSBMmixed<double>(m,n,dV,dlev);

    std::cout << "Orthogonality test:  " << orthogonality_test<double>(m,n,dV,n) << '\n';
    std::cout << "Leverage Score test: " << levscore_test<double>(m,n,dV,dlev) << '\n';

    delete[] dV;
    delete[] dlev;

    return 0;
}

template<typename T>
void OSBM(int64_t n_rows, int64_t n_cols, T (*V), T (*lev)) {
    int its = 0;               /* Records number of iterations to convergence */
    int a,b,c;                 /* Indexing variables */
    int i,j;                   /* Stores indices of rows that satisfies majorization condition */
    bool cond, ccond;          /* cond indicates if the majorization condition has found two indices i,j
                                  ccond helps to break the loop if an innerloop is not satisfied  */
    T r_ii, r_jj, r_ij;        /* values used to compute cos and sin for the givens rotation */
    T t, cos, sin;           
    T rownorms[n_rows];        /* Array to hold row norms of V at each iteration */

    if (check_levscores<T>(n_rows, n_cols, lev) == -1){
        throw std::invalid_argument("Leverage scores are invalid");
    }

    i = n_rows - n_cols - 1;
    j = n_rows - n_cols;

    for (a = 0; a < n_rows; a++){
        rownorms[a] = 0.0;
    }
    for (a = n_rows-n_cols; a < n_rows; a++) { 
        rownorms[a] = 1.0;
    }

    while(true) {
        cond = false;
        ccond = true;

        if (its==0 && check_majorization<T>(n_rows, n_cols, rownorms, lev)==-1){
            throw std::invalid_argument("Matrix row norms do not majorize the leverage scores");
        }

        if (its==n_rows-1) { 
            break;
        }

        r_ii = rownorms[i];
        r_jj = rownorms[j];
        r_ij = blas::dot(n_cols, &(V)[i*n_cols], 1, &(V)[j*n_cols], 1);

        if ((lev)[i] - r_ii < r_jj - (lev)[j]) {
            t = (sgn<T>(r_ij)*r_ij + sqrt(pow(r_ij,2) - (r_ii - (lev)[i])*(r_jj - (lev)[i]))) / (r_jj - (lev)[i]);
            cos = 1/sqrt(1 + pow(t,2));
            sin = cos*t;
            blas::rot(n_cols, &(V)[i*n_cols], 1, &(V)[j*n_cols],1, cos, sin);
            rownorms[i] = lev[i];
            rownorms[j] = blas::dot(n_cols, &(V)[j*n_cols], 1, &(V)[j*n_cols], 1);
            i = i-1;
        } else {
            t = (-sgn<T>(r_ij)*r_ij - sqrt(pow(r_ij,2) - (r_ii -  (lev)[j])*(r_jj - (lev)[j]))) / (r_ii - (lev)[j]);
            cos = 1/sqrt(1 + pow(t,2));
            sin = cos*t;
            blas::rot(n_cols, &(V)[i*n_cols], 1, &(V)[j*n_cols],1, cos, sin);
            rownorms[j] = lev[j];
            rownorms[i] = blas::dot(n_cols, &(V)[i*n_cols], 1, &(V)[i*n_cols], 1);
            j += 1;
        }

        its += 1;
        cond = false;

    }
}

template<typename T>
void OSBMmixed(int64_t n_rows, int64_t n_cols, T (*V), T (*lev)) {
    int its = 0;               /* Records number of iterations to convergence */
    int a,b,c;                 /* Indexing variables */
    int i,j;                   /* Stores indices of rows that satisfies majorization condition */
    bool cond, ccond;          /* cond indicates if the majorization condition has found two indices i,j
                                  ccond helps to break the loop if an innerloop is not satisfied  */
    long double r_ii, r_jj, r_ij;        /* values used to compute cos and sin for the givens rotation */
    long double t, cos, sin;           
    T rownorms[n_rows];        /* Array to hold row norms of V at each iteration */

    if (check_levscores<T>(n_rows, n_cols, lev) == -1){
        throw std::invalid_argument("Leverage scores are invalid");
    }

    i = n_rows - n_cols - 1;
    j = n_rows - n_cols;

    for (a = 0; a < n_rows; a++){
        rownorms[a] = 0.0;
    }
    for (a = n_rows-n_cols; a < n_rows; a++) { 
        rownorms[a] = 1.0;
    }

    while(true) {
        cond = false;
        ccond = true;

        if (its==0 && check_majorization<T>(n_rows, n_cols, rownorms, lev)==-1){
            throw std::invalid_argument("Matrix row norms do not majorize the leverage scores");
        }

        if (its==n_rows-1) { 
            break;
        }

        r_ii = (long double)rownorms[i];
        r_jj = (long double)rownorms[j];
        r_ij = (long double)blas::dot(n_cols, &(V)[i*n_cols], 1, &(V)[j*n_cols], 1);

        if ((lev)[i] - r_ii < r_jj - (lev)[j]) {
            t = (sgn<long double>(r_ij)*r_ij + sqrt(pow(r_ij,2) - (r_ii - (long double)(lev)[i])*(r_jj - (long double)(lev)[i]))) 
                / (r_jj - (long double)(lev)[i]);
            cos = 1/sqrt(1 + pow(t,2));
            sin = cos*t;
            blas::rot(n_cols, &(V)[i*n_cols], 1, &(V)[j*n_cols],1, (T)cos, (T)sin);
            rownorms[i] = lev[i];
            rownorms[j] = blas::dot(n_cols, &(V)[j*n_cols], 1, &(V)[j*n_cols], 1);
            i = i-1;
        } else {
            t = (-sgn<long double>(r_ij)*r_ij - sqrt(pow(r_ij,2) - (r_ii -  (long double)(lev)[j])*(r_jj - (long double)(lev)[j]))) 
                / (r_ii - (long double)(lev)[j]);
            cos = 1/sqrt(1 + pow(t,2));
            sin = cos*t;
            blas::rot(n_cols, &(V)[i*n_cols], 1, &(V)[j*n_cols],1, (T)cos, (T)sin);
            rownorms[j] = lev[j];
            rownorms[i] = blas::dot(n_cols, &(V)[i*n_cols], 1, &(V)[i*n_cols], 1);
            j += 1;
        }

        its += 1;
        cond = false;

    }
}

template<typename T>
T levscore_test(int64_t n_rows, int64_t n_cols, T (*V), T (*lev)) {
    T max = abs((lev)[0] - blas::dot(n_cols, V, 1, V, 1));
    T temp;
    for (int i=1; i<n_rows; i++) {
        temp = abs((lev)[i] - blas::dot(n_cols, &(V)[i*n_cols], 1, &(V)[i*n_cols], 1));
        if (temp > max) {
            max = temp;
        }
    }
    return max;
}

template<typename T>
T orthogonality_test(int64_t n_rows, int64_t n_cols, T (*V), int64_t ldc) {
    int i,j;
    T norm = 0;
    T *C = new T[n_cols*n_cols];
    blas::gemm(blas::Layout::RowMajor, blas::Op::Trans, blas::Op::NoTrans, n_cols, n_cols, n_rows, 1, V, ldc, V, ldc, 0, C, n_cols);
    for (i=0; i<n_cols; i++){
        C[i*n_cols + i] -= 1;
    }
    for (i=0; i<n_cols*n_cols; i++){
        norm += pow(C[i],2);
    }
    delete[] C;
    return sqrt(norm);
}

template<typename T>
int check_levscores(int64_t n_rows, int64_t n_cols, T (*lev)) {
    T sum = 0;
    for (int i=0; i<n_rows; i++) {
        if (lev[i]<0 || lev[i]>1) {
            std::cout << "A leverage score is out of bounds" << '\n';
            return -1;
        }
        sum += lev[i];
    }
    std::cout << ""; 
    if (abs(sum - n_cols) > std::numeric_limits<T>::epsilon()*n_rows*n_cols) {
        std::cout << "Sum of leverage scores do not add up to n_cols:" << abs(sum-n_cols) << '\n';
        return -1;
    }
    return 0;
}

template<typename T>
int check_majorization(int64_t n_rows, int64_t n_cols, T *rownorms, T *lev) {
    T sum_rownorms = 0;
    T sum_lev = 0;
    for (int i=n_rows-1; i>0; i+=-1) {
        sum_rownorms += rownorms[i];
        sum_lev += lev[i];
        //if (sum_rownorms - sum_lev < std::numeric_limits<T>::epsilon()*n_cols) {
        if (sum_rownorms < sum_lev) {
            return -1;
        }
    }
    sum_rownorms += rownorms[0];
    sum_lev += lev[0];
    return (abs(sum_rownorms - sum_lev) < std::numeric_limits<T>::epsilon()*n_cols);
}

template <typename T> 
T sgn(T val) {
    return (T(0) < val) - (val < T(0));
}
