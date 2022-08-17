#include <blas.hh>
#include <lapack.hh>
#include <RandBLAS.hh>

#include <math.h>
#include <unistd.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h>

template<typename T, size_t rows, size_t cols>
void print_mat(double M[rows][cols]); 

template <typename T> int sgn(T val);

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
                if (((*lev)[a] - colnorms[a] > 1e-10) && (colnorms[b] - (*lev)[b] > 1e-10)) {
                    ccond = true;
                    for (c=a+1; c<b; c++) {
                        if (abs(colnorms[c] - (*lev)[c]) > 1e-12) {
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


int main( int argc, char *argv[] ) {
    double V[3][5] = {{0,0,1,0,0}, {0,0,0,1,0}, {0,0,0,0,1}};
    double ell[5] = {0.4, 0.5, 0.6, 0.7, 0.8};
    OSBM<double,3,5>(&V, &ell);
    print_mat<double,3,5>(V);
}


template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
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
