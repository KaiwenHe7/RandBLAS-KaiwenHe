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

template<typename T, size_t rows, size_t cols>
void print_mat(double M[rows][cols]); 

template <typename T> int sgn(T val);

template<typename T, int64_t n_rows, int64_t n_cols>
void OSBM(T (*V)[n_rows][n_cols], T (*lev)[n_cols]);

template<typename T, int64_t n_rows, int64_t n_cols>
void OSBMrm(T (*V)[n_rows*n_cols], T (*lev)[n_rows]);

template<typename T, int64_t n_rows, int64_t n_cols>
T levscore_test(T (*V)[n_rows*n_cols], T (*lev)[n_rows]);

template<typename T>
T levscore_test(int64_t n_rows, int64_t n_cols, T (*V), T (*lev));

template<typename T>
T orthogonality_test(int64_t n_rows, int64_t n_cols, T (*V));

template<typename T>
void OSBMrmt(int64_t n_rows, int64_t n_cols, T (*V), T (*lev));

template<typename T>
int check_levscores(int64_t n_rows, int64_t n_cols, T (*lev));

template<typename T>
int check_majorization(int64_t n_rows, int64_t n_cols, T (*V), T(*lev));

int main( int argc, char *argv[] ) {
    double V[3][5] = {{0,0,1,0,0}, {0,0,0,1,0}, {0,0,0,0,1}};
    double Vrm[15] = {0,0,0, 0,0,0, 1,0,0, 0,1,0, 0,0,1};
    double ell[5] = {0.4, 0.5, 0.6, 0.6, 0.9};
    OSBM<double,3,5>(&V, &ell);
    OSBMrmt<double>(5,3,Vrm,ell);
    for (int i=0; i<15; i++) {
        std::cout << Vrm[i] << '\n';
    }
    print_mat<double,3,5>(V);
    std::cout << '\n';
    std::cout << "Max lev score diff:  " << levscore_test<double>(5,3,Vrm, ell) << '\n';
    std::cout << "Orthogonality test:  " << orthogonality_test<double>(5,3,Vrm) << '\n';


}


template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

template<typename T, int64_t n_rows, int64_t n_cols>
void OSBM(T (*V)[n_rows][n_cols], T (*lev)[n_cols]) {
    int its = 0;               /* Records number of iterations to convergence */
    int a,b,c,i,j;             /* Indexing variables */
    bool cond, ccond;          /* cond indicates if the majorization condition has found two indices i,j
                                  ccond helps to break the loop if an innerloop is not satisfied  */
    T r_ii, r_jj, r_ij;     /* values used to compute cos and sin for the givens rotation */
    T t, cos, sin;           
    T colnorms[n_cols];     /* Array to hold row norms of V at each iteration */
    T coltemp[n_rows], coltemp2[n_rows]; /* Temporary arrays to store col of V */
    
    while(true) {
        cond = false;
        ccond = true;

        for (a=0; a<n_cols; a++){      /* updates colnorms to store col norms of V at each iteration */
            for (b=0; b<n_rows; b++){
                coltemp[b] = (*V)[b][a];
            }
            colnorms[a] = blas::dot(n_rows, coltemp, 1, coltemp, 1);
        }


        for (a=0; a<(n_cols-1); a++) {    /* OSBM conditional to choose indices i,j to perform givens rotations on */
            for (b=a+1; b<n_cols; b++) {
                if (((*lev)[a] - colnorms[a] > std::numeric_limits<double>::epsilon()*n_cols) && (colnorms[b] - (*lev)[b] > std::numeric_limits<double>::epsilon()*n_cols)) {
                    ccond = true;
                    for (c=a+1; c<b; c++) {
                        if (abs(colnorms[c] - (*lev)[c]) > std::numeric_limits<double>::epsilon()*n_cols) {
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

        if (cond==false || its==n_cols) {
            break;
        }

        for (a=0;a<n_rows;a++) {         /*stores col i and col j to perform dot product on for r_ij*/
            coltemp[a] = (*V)[a][i];
            coltemp2[a] = (*V)[a][j];
        }  

        r_ii = colnorms[i];
        r_jj = colnorms[j];
        r_ij = blas::dot(n_rows, coltemp, 1, coltemp2, 1);

        if ((*lev)[i] - r_ii < r_jj - (*lev)[j]) {
            t = (r_ij + sqrt(pow(r_ij,2) - (r_ii - (*lev)[i])*(r_jj - (*lev)[i]))) / (r_jj - (*lev)[i]);
            cos = 1/sqrt(1 + pow(t,2));
            sin = cos*t;
            blas::rot(n_rows, coltemp, 1, coltemp2, 1, cos, sin);
            for (a=0;a<n_rows;a++){
                (*V)[a][i] = coltemp[a];
                (*V)[a][j] = coltemp2[a];
            }
        } else {
            t = (-r_ij - sqrt(pow(r_ij,2) - (r_ii -  (*lev)[j])*(r_jj - (*lev)[j]))) / (r_ii - (*lev)[j]);
            cos = 1/sqrt(1 + pow(t,2));
            sin = cos*t;
            blas::rot(n_rows, coltemp, 1, coltemp2, 1, cos, sin);
            for (a=0;a<n_rows;a++){
                (*V)[a][i] = coltemp[a];
                (*V)[a][j] = coltemp2[a];
            }
        }

        its += 1;
        cond = false;

    }
}

/*TODO: Write tests in proper testing framwork: gtest */

template<typename T, int64_t n_rows, int64_t n_cols>
void OSBMrm(T (*V)[n_rows*n_cols], T (*lev)[n_rows]) {
    int its = 0;               /* Records number of iterations to convergence */
    int a,b,c;                 /* Indexing variables */
    int i,j;                   /* Stores indices of rows that satisfies majorization condition */
    bool cond, ccond;          /* cond indicates if the majorization condition has found two indices i,j
                                  ccond helps to break the loop if an innerloop is not satisfied  */
    T r_ii, r_jj, r_ij;        /* values used to compute cos and sin for the givens rotation */
    T t, cos, sin;           
    T rownorms[n_rows];        /* Array to hold row norms of V at each iteration */
    
    while(true) {
        cond = false;
        ccond = true;

        for (a=0; a<n_rows; a++){      /* updates colnorms to store col norms of V at each iteration */
            rownorms[a] = blas::dot(n_cols, &(*V)[a*n_cols], 1, &(*V)[a*n_cols], 1);
        }

        for (a=0; a<(n_rows-1); a++) {    
            for (b=a+1; b<n_rows; b++) {
                if (((*lev)[a] - rownorms[a] > std::numeric_limits<double>::epsilon()*n_cols) && (rownorms[b] - (*lev)[b] > std::numeric_limits<double>::epsilon()*n_cols)) {
                    ccond = true;
                    for (c=a+1; c<b; c++) {
                        if (abs(rownorms[c] - (*lev)[c]) > std::numeric_limits<double>::epsilon()*n_cols) {
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
        r_ij = blas::dot(n_cols, &(*V)[i*n_cols], 1, &(*V)[j*n_cols], 1);

        if ((*lev)[i] - r_ii < r_jj - (*lev)[j]) {
            t = (r_ij + sqrt(pow(r_ij,2) - (r_ii - (*lev)[i])*(r_jj - (*lev)[i]))) / (r_jj - (*lev)[i]);
            cos = 1/sqrt(1 + pow(t,2));
            sin = cos*t;
            blas::rot(n_cols, &(*V)[i*n_cols], 1, &(*V)[j*n_cols],1, cos, sin);
        } else {
            t = (-r_ij - sqrt(pow(r_ij,2) - (r_ii -  (*lev)[j])*(r_jj - (*lev)[j]))) / (r_ii - (*lev)[j]);
            cos = 1/sqrt(1 + pow(t,2));
            sin = cos*t;
            blas::rot(n_cols, &(*V)[i*n_cols], 1, &(*V)[j*n_cols],1, cos, sin);
        }

        its += 1;
        cond = false;

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
    
    while(true) {
        cond = false;
        ccond = true;

        for (a=0; a<n_rows; a++){      /* updates colnorms to store col norms of V at each iteration */
            rownorms[a] = blas::dot(n_cols, &(V)[a*n_cols], 1, &(V)[a*n_cols], 1);
        }

        for (a=0; a<(n_rows-1); a++) {    
            for (b=a+1; b<n_rows; b++) {
                if (((lev)[a] - rownorms[a] > std::numeric_limits<double>::epsilon()*n_cols) && (rownorms[b] - (lev)[b] > std::numeric_limits<double>::epsilon()*n_cols)) {
                    ccond = true;
                    for (c=a+1; c<b; c++) {
                        if (abs(rownorms[c] - (lev)[c]) > std::numeric_limits<double>::epsilon()*n_cols) {
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
            t = (r_ij + sqrt(pow(r_ij,2) - (r_ii - (lev)[i])*(r_jj - (lev)[i]))) / (r_jj - (lev)[i]);
            cos = 1/sqrt(1 + pow(t,2));
            sin = cos*t;
            blas::rot(n_cols, &(V)[i*n_cols], 1, &(V)[j*n_cols],1, cos, sin);
        } else {
            t = (-r_ij - sqrt(pow(r_ij,2) - (r_ii -  (lev)[j])*(r_jj - (lev)[j]))) / (r_ii - (lev)[j]);
            cos = 1/sqrt(1 + pow(t,2));
            sin = cos*t;
            blas::rot(n_cols, &(V)[i*n_cols], 1, &(V)[j*n_cols],1, cos, sin);
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
    return norm;
}

template<typename T>
int check_levscores(int64_t n_rows, int64_t n_cols, T (*lev)) {
    T sum;
    for (int i=0; i<n_rows; i++) {
        if (lev[i]<0 || lev[i]>1) {
            return -1;
        }
        sum += lev[i];
    }
    if (abs(sum - (T) n_cols) > std::numeric_limits<double>::epsilon()*n_cols) {
        return -1;
    }
    return 0;
}

template<typename T>
int check_majorization(int64_t n_rows, int64_t n_cols, T (*rownorms), T(*lev)) {
    T sum_rownorms = 0;
    T sum_lev = 0;
    for (int i=n_rows-1; i>-1; i+=-1) {
        sum_rownorms += rownorms[i];
        sum_lev += lev[i];
        if (sum_rownorms < sum_lev) {
            return -1;
        }
    }
    return (abs(sum_rownorms - sum_lev) < std::numeric_limits<double>::epsilon()*n_cols);
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
