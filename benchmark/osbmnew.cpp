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

using namespace RandBLAS::dense_op;

template<typename T, size_t rows, size_t cols>
void print_mat(double M[rows][cols]); 

template <typename T> T sgn(T val);

template<typename T, int64_t n_rows, int64_t n_cols>
T levscore_test(T (*V)[n_rows*n_cols], T (*lev)[n_rows]);

template<typename T>
T levscore_test(int64_t n_rows, int64_t n_cols, T (*V), T (*lev));

template<typename T>
T orthogonality_test(int64_t n_rows, int64_t n_cols, T (*V));

template<typename T>
void OSBM(int64_t n_rows, int64_t n_cols, T (*V), T (*lev));

template<typename T>
int check_levscores(int64_t n_rows, int64_t n_cols, T (*lev));

template<typename T>
int check_majorization(int64_t n_rows, int64_t n_cols, T (*rownorms), T (*lev));

template<typename T>
void random_sample(int64_t n_rows, int64_t n_cols, T (*V), uint32_t seed);

template<typename T>
int temp_test();

template<typename T>
void OSBMrmt(int64_t n_rows, int64_t n_cols, T (*V), T (*lev));

template<typename T>
void OSBMbs(int64_t n_rows, int64_t n_cols, T (*V), T (*lev));

template<typename T>
void gen_osbm(T *V, T *lev, int64_t n_rows, int64_t n_cols);

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

int main( int argc, char *argv[] ) {
    int64_t n_rows = 50;
    int64_t n_cols = 10;
    int i;
    double V[n_rows*n_cols];
    double lev[n_rows];
    //double lev[6] = {0.1, 0.4, 0.5, 0.6, 0.6, 0.8};
    gen_rvec_lev<double>(lev, n_rows, n_cols);
    std::fill(V, V+n_rows*n_cols, 0);
    for (i=0; i<n_cols; i++) {
        V[(n_rows-n_cols)*n_cols + i + i*n_cols] = 1;
    }
    OSBMrmt<double>(n_rows,n_cols,V,lev);

    /*for (i=0; i < 15; i++) {
        std::cout << V[i] << '\n';
    }*/
    std::cout << "---------------" << '\n';
    std::cout << orthogonality_test(n_rows,n_cols,V) << '\n';
    std::cout << levscore_test(n_rows,n_cols,V,lev) << '\n';
    return 0;
}

template<typename T>
void OSBMbs(int64_t n_rows, int64_t n_cols, T (*V), T (*lev)) {
    int its = 0;               /* Records number of iterations to convergence */
    int a;
    int i,j;                   /* Stores indices of rows that satisfies majorization condition */
    bool breakcond = false;          /* cond indicates if the majorization condition has found two indices i,j
                                  ccond helps to break the loop if an innerloop is not satisfied  */
    T r_ii, r_jj, r_ij;        /* values used to compute cos and sin for the givens rotation */
    T t, cos, sin;           
    T rownorms[n_rows];        /* Array to hold row norms of V at each iteration */

    if (check_levscores<T>(n_rows, n_cols, lev) == -1){
        throw std::invalid_argument("Leverage scores are invalid");
    }
    
    std::fill(rownorms, rownorms+n_rows, 0);
    for (a = n_cols-1; a < n_rows; a++) {
        rownorms[a] = 1;
    }

    if (its==0 && check_majorization<T>(n_rows, n_cols, rownorms, lev)==-1){
        throw std::invalid_argument("Matrix row norms do not majorize the leverage scores");
    }

    i = n_rows - n_cols - 1;
    j = n_rows - n_cols;

    std::cout << "(" << i << ", " << j << ", " << 1 << ")" << '\n';
    for (its = 0; its<n_rows-1; its++) {

        if (i == 0 || j==(n_rows-1)) {
            breakcond = true;
        }

        r_ii = rownorms[i];      // dot(row[i], row[i])
        r_jj = rownorms[j];      // dot(row[j], row[j])
        r_ij = blas::dot(n_cols, &(V)[i*n_cols], 1, &(V)[j*n_cols], 1);    //dot(row[i], row[j])

        if ((lev)[i] - r_ii < r_jj - (lev)[j]) {
            std::cout << "CASE 1" << '\n';
            std::cout << "(" << i << ", " << j << ", " << 1 << ")" << '\n';
            t = (sgn<T>(r_ij)*r_ij + sqrt(pow(r_ij,2) - (r_ii - (lev)[i])*(r_jj - (lev)[i]))) / (r_jj - (lev)[i]);
            cos = 1/sqrt(1 + pow(t,2));
            sin = cos*t;
            blas::rot(n_cols, &(V)[i*n_cols], 1, &(V)[j*n_cols],1, cos, sin);
            rownorms[i] = lev[i];
            rownorms[j] = blas::dot(n_cols, &(V)[j*n_cols], 1, &(V)[j*n_cols], 1);
            i = i - 1;
        } else {
            std::cout << "CASE 2" << '\n';
            std::cout << "(" << i << ", " << j << ", " << 2 << ")" << '\n';
            t = (-sgn<T>(r_ij)*r_ij - sqrt(pow(r_ij,2) - (r_ii -  (lev)[j])*(r_jj - (lev)[j]))) / (r_ii - (lev)[j]);
            cos = 1/sqrt(1 + pow(t,2));
            sin = cos*t;
            blas::rot(n_cols, &(V)[i*n_cols], 1, &(V)[j*n_cols],1, cos, sin);
            rownorms[i] = blas::dot(n_cols, &(V)[i*n_cols], 1, &(V)[i*n_cols], 1);
            rownorms[j] = lev[j];
            j = j + 1;
        }

        if (breakcond) {
            break;
        }
        
    }
}

template<typename T>
void OSBMrmt(int64_t n_rows, int64_t n_cols, T (*V), T (*lev)) {
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
            std::cout << "(" << i << ", " << j << ", " << 1 << ")" << '\n';
            t = (r_ij + sqrt(pow(r_ij,2) - (r_ii - (lev)[i])*(r_jj - (lev)[i]))) / (r_jj - (lev)[i]);
            cos = 1/sqrt(1 + pow(t,2));
            sin = cos*t;
            blas::rot(n_cols, &(V)[i*n_cols], 1, &(V)[j*n_cols],1, cos, sin);
            rownorms[i] = lev[i];
            rownorms[j] = blas::dot(n_cols, &(V)[j*n_cols], 1, &(V)[j*n_cols], 1);
            i = i-1;
        } else {
            std::cout << "(" << i << ", " << j << ", " << 2 << ")" << '\n';
            t = (-r_ij - sqrt(pow(r_ij,2) - (r_ii -  (lev)[j])*(r_jj - (lev)[j]))) / (r_ii - (lev)[j]);
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
void OSBM(int64_t n_rows, int64_t n_cols, T (*V), T (*lev)) {
    int its = 0;               /* Records number of iterations to convergence */
    int a,b,c;                 /* Indexing variables */
    int i,j;                   /* Stores indices of rows that satisfies majorization condition */
    bool breakcond, nextcond;          /* cond indicates if the majorization condition has found two indices i,j
                                  ccond helps to break the loop if an innerloop is not satisfied  */
    T r_ii, r_jj, r_ij;        /* values used to compute cos and sin for the givens rotation */
    T t, cos, sin;           
    T rownorms[n_rows];        /* Array to hold row norms of V at each iteration */

    if (check_levscores<T>(n_rows, n_cols, lev) == -1){
        throw std::invalid_argument("Leverage scores are invalid");
    }
    
    std::fill(rownorms, rownorms+n_rows, 0);
    for (a = n_cols; a < n_rows; a++) {
        rownorms[a] = 1;
    }
    
    while(true) {
        breakcond = false;  // If no pair i<j is found then we are done. Set breakcond = true
        nextcond = true;    // If a pair i<j does not satisfy the condition then proceed for next i

        if (its==0 && check_majorization<T>(n_rows, n_cols, rownorms, lev)==-1){
            throw std::invalid_argument("Matrix row norms do not majorize the leverage scores");
        }

        for (a=0; a<(n_rows-1); a++) {    
            for (b=a+1; b<n_rows; b++) {
                if (((lev)[a] - rownorms[a] > std::numeric_limits<T>::epsilon()*2) 
                        && (rownorms[b] - (lev)[b] > std::numeric_limits<T>::epsilon()*2)) {
                    nextcond = true;
                    for (c=a+1; c<b; c++) {
                        if (abs(rownorms[c] - (lev)[c]) > std::numeric_limits<T>::epsilon()*2) {
                            nextcond = false;
                            break;
                        }
                    }
                    if (nextcond == true) {
                        i = a;
                        j = b;
                        breakcond = true;
                        break;
                    }
                }
                
            }
            if (breakcond == true) {
                break;
            }    
        } 

        if (breakcond==false || its==n_rows) { 
            break;
        }

        r_ii = rownorms[i];      // dot(row[i], row[i])
        r_jj = rownorms[j];      // dot(row[j], row[j])
        r_ij = blas::dot(n_cols, &(V)[i*n_cols], 1, &(V)[j*n_cols], 1);    //dot(row[i], row[j])

        if ((lev)[i] - r_ii < r_jj - (lev)[j]) {
            std::cout << "(" << i << ", " << j << ", " << 1 << ")" << '\n';
            t = (sgn<T>(r_ij)*r_ij + sqrt(pow(r_ij,2) - (r_ii - (lev)[i])*(r_jj - (lev)[i]))) / (r_jj - (lev)[i]);
            cos = 1/sqrt(1 + pow(t,2));
            sin = cos*t;
            blas::rot(n_cols, &(V)[i*n_cols], 1, &(V)[j*n_cols],1, cos, sin);
            rownorms[i] = lev[i];
            rownorms[j] = blas::dot(n_cols, &(V)[j*n_cols], 1, &(V)[j*n_cols], 1);
        } else {
            std::cout << "(" << i << ", " << j << ", " << 2 << ")" << '\n';
            t = (-sgn<T>(r_ij)*r_ij - sqrt(pow(r_ij,2) - (r_ii -  (lev)[j])*(r_jj - (lev)[j]))) / (r_ii - (lev)[j]);
            cos = 1/sqrt(1 + pow(t,2));
            sin = cos*t;
            blas::rot(n_cols, &(V)[i*n_cols], 1, &(V)[j*n_cols],1, cos, sin);
            rownorms[i] = blas::dot(n_cols, &(V)[i*n_cols], 1, &(V)[i*n_cols], 1);
            rownorms[j] = lev[j];
        }

        its += 1;
        breakcond = false;

    }
}

template<typename T>
void gen_osbm(T *V, T *lev, int64_t n_rows, int64_t n_cols) {
    int i;
    std::fill(V, V+n_rows*n_cols, 0);
    std::cout << "Filled V" << '\n';
    for (i=0; i<n_cols; i++) {
        V[(n_rows-n_cols)*n_cols + i + i*n_cols] = 1;
    }
    std::cout << "V now [o, I]" << '\n';
    T sum = 0;
    RandBLAS::dense_op::gen_rmat_unif<double>(n_rows, 1, lev, 1);
    std::cout << lev[n_rows-2] << '\n';
    blas::scal(n_rows, 0.5, lev, 1); 
    for (i=0; i<n_rows; i++) {
        lev[i] += 0.5;
        sum += lev[i];
    }
    blas::scal(n_rows, n_cols/sum, lev, 1);
    std::sort(lev, lev+n_rows);
    RandBLAS::osbm::OSBM<double>(n_rows, n_cols, V, lev);

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
T orthogonality_test(int64_t n_rows, int64_t n_cols, T (*V)) {
    int i,j;
    T norm = 0;
    T C[n_cols*n_cols];
    blas::gemm(blas::Layout::RowMajor, blas::Op::Trans, blas::Op::NoTrans, n_cols, n_cols, n_rows, 1, V, n_cols, V, n_cols, 0, C, n_cols);
    for (i=0; i<n_cols; i++){
        C[i*n_cols + i] -= 1;
    }
    for (i=0; i<n_cols*n_cols; i++){
        norm += pow(C[i],2);
    }
    return sqrt(norm);
}
//TODO: Run tests for cases where leverage scores are close to the boundary
//TODO: Message Max or Parth on slack regarding gtesting errors
//
template<typename T>
int check_levscores(int64_t n_rows, int64_t n_cols, T (*lev)) {
    T sum = 0;
    for (int i=0; i<n_rows; i++) {
        if (lev[i]<=0 || lev[i]>=1) {
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
int check_majorization(int64_t n_rows, int64_t n_cols, T (*rownorms), T(*lev)) {
    T sum_rownorms = 0;
    T sum_lev = 0;
    for (int i=n_rows-1; i>0; i+=-1) {
        sum_rownorms += rownorms[i];
        sum_lev += lev[i];
        if (sum_rownorms - sum_lev < std::numeric_limits<T>::epsilon()*n_cols) {
            return -1;
        }
    }
    sum_rownorms += rownorms[0];
    sum_lev += lev[0];
    return (abs(sum_rownorms - sum_lev) < std::numeric_limits<T>::epsilon()*n_cols);
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
template<typename T>
void random_sample(int64_t n_rows, int64_t n_cols, T (*V), uint32_t seed) {
    int i,j;
    T signu0;              // Stores the sign of u[0] to avoid cancellation
    T u[n_cols];           // vector u to store random normal values. Used in the construction of Householder.
    T norm;                // Stores the norm of u to normalize u
    for (i=n_cols-1; i>0; i--) {
        RandBLAS::dense_op::gen_rmat_norm<T>(1, i+1, u, seed);  
        signu0 = sgn<T>(u[0]);                  
        u[0] = u[0] + signu0 * blas::nrm2(i+1,u,1);
        norm = blas::nrm2(i+1, u, 1);
        blas::scal(i+1, 1/norm, u, 1);
        lapack::larf(lapack::Side::Left, i+1, n_rows, &u[0], 1, 2, &V[n_cols-i-1], n_cols); 
        blas::scal(n_rows, signu0, &V[n_cols-i+1], n_cols);
    }
}
//TODO: 1000 rows 100 cols
template <typename T> T sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

/*            std::cout << "CASE 1-----------------------------" << '\n';
            std::cout << "its " << its << ':' << '\n';
            std::cout << "       " << "row " << i << ":" << "  rownorms[" << i << "]: " << rownorms[i] << ";  Lev[" << i << "]:" << lev[i] << '\n';     
            std::cout << "       " << "row " << j << ":" << "  rownorms[" << j << "]: " << rownorms[j] << ";  Lev[" << j << "]:" << lev[j] << '\n';     

            std::cout << "CASE 2-----------------------------" << '\n';
            std::cout << "its " << its << ':' << '\n';
            std::cout << "       " << "row " << i << ":" << "  rownorms[" << i << "]: " << rownorms[i] << ";  Lev[" << i << "]:" << lev[i] << '\n';     
            std::cout << "       " << "row " << j << ":" << "  rownorms[" << j << "]: " << rownorms[j] << ";  Lev[" << j << "]:" << lev[j] << '\n';     */
