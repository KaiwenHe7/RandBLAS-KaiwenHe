#include "osbm.hh"
#include <unistd.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>

namespace RandBLAS::osbm {

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

    while(true) {
        cond = false;
        ccond = true;

        for (a=0; a<n_rows; a++){      /* updates colnorms to store col norms of V at each iteration */
            rownorms[a] = blas::dot(n_cols, &(V)[a*n_cols], 1, &(V)[a*n_cols], 1);
        }
        
        if (its==0 && check_majorization<T>(n_rows, n_cols, rownorms, lev)==-1){
            throw std::invalid_argument("Matrix row norms do not majorize the leverage scores");
        }

        for (a=0; a<(n_rows-1); a++) {    
            for (b=a+1; b<n_rows; b++) {
                if (((lev)[a] - rownorms[a] > std::numeric_limits<T>::epsilon()*n_cols) && (rownorms[b] - (lev)[b] > std::numeric_limits<T>::epsilon()*n_cols)) {
                    ccond = true;
                    for (c=a+1; c<b; c++) {
                        if (abs(rownorms[c] - (lev)[c]) > std::numeric_limits<T>::epsilon()*n_cols) {
                            ccond = false;
                            break;
                        }
                    }
                    if (ccond == true) {
                        i = a;
                        j = b;
                        cond = true;
                        break;
                    }
                }
                
            }
            if (cond == true) {
                break;
            }    
        } 

        if (cond==false || its==n_rows) { 
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
        } else {
            t = (-sgn<T>(r_ij)*r_ij - sqrt(pow(r_ij,2) - (r_ii -  (lev)[j])*(r_jj - (lev)[j]))) / (r_ii - (lev)[j]);
            cos = 1/sqrt(1 + pow(t,2));
            sin = cos*t;
            blas::rot(n_cols, &(V)[i*n_cols], 1, &(V)[j*n_cols],1, cos, sin);
        }

        its += 1;
        cond = false;

    }
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
    if (abs(sum - n_cols) > std::numeric_limits<T>::epsilon()*n_rows) {
        std::cout << "Sum of leverage scores do not add up to n_cols" << '\n';
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
    T C[n_cols*n_cols];
    blas::gemm(blas::Layout::RowMajor, blas::Op::Trans, blas::Op::NoTrans, n_cols, n_cols, n_rows, 1, V, ldc, V, ldc, 0, C, n_cols);
    for (i=0; i<n_cols; i++){
        C[i*n_cols + i] -= 1;
    }
    for (i=0; i<n_cols*n_cols; i++){
        norm += pow(C[i],2);
    }
    return sqrt(norm);
}

template <typename T> 
T sgn(T val) {
    return (T(0) < val) - (val < T(0));
}


template float RandBLAS::osbm::sgn<float>(float val);
template double RandBLAS::osbm::sgn<double>(double val);

template float RandBLAS::osbm::orthogonality_test<float>(int64_t n_rows, int64_t n_cols, float *V, int64_t ldc);
template double RandBLAS::osbm::orthogonality_test<double>(int64_t n_rows, int64_t n_cols, double *V, int64_t ldc);

template float RandBLAS::osbm::levscore_test<float>(int64_t n_rows, int64_t n_cols, float *V, float *lev);
template double RandBLAS::osbm::levscore_test<double>(int64_t n_rows, int64_t n_cols, double *V, double *lev);

template int RandBLAS::osbm::check_levscores<float>(int64_t n_rows, int64_t n_cols, float *lev);
template int RandBLAS::osbm::check_levscores<double>(int64_t n_rows, int64_t n_cols, double *lev);

template int RandBLAS::osbm::check_majorization<float>(int64_t n_rows, int64_t n_cols, float *rownorms, float *lev);
template int RandBLAS::osbm::check_majorization<double>(int64_t n_rows, int64_t n_cols, double *rownorms, double *lev);

template void RandBLAS::osbm::OSBM<float>(int64_t n_rows, int64_t n_cols, float *V, float *lev);
template void RandBLAS::osbm::OSBM<double>(int64_t n_rows, int64_t n_cols, double *V, double *lev);

}
