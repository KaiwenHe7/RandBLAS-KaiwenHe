#include <blas.hh>
#include <lapack.hh>
#include <RandBLAS.hh>

#include <math.h>
#include <unistd.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h>

template<typename T, int64_t n_rows, int64_t n_cols>
void OSBM(T (*V)[n_rows][n_cols], T (*lev)[n_cols]) {
    int its = 0;               /* Records number of iterations to convergence */
    int a,b,c,i,j;             /* Indexing variables */
    bool cond, ccond;          /* cond indicates if the majorization condition has found two indices i,j
                                  ccond helps to break the loop if an innerloop is not satisfied  */
    T r_ii, r_jj, r_ij;     /* values used to compute cos and sin for the givens rotation */
    T t, cos, sin;           
    T colnorms[n_cols];     /* Array to hold row norms of V at each iteration */
    T coltemp[n_rows];
    
    while(true) {
        cond = false;
        ccond = true;

        for (a=0; a<n_cols; a++){
            for (b=0; b<n_rows; b++){
                coltemp[b] = (*V)[b][a];
            }
            colnorms[a] = blas::dot(n_rows, coltemp, 1, coltemp, 1);
        }


        for (a=0; a<(n_cols-1); a++) {
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
       

        std::cout << i << "   " << j << '\n';
        break; 
    }
}


int main( int argc, char *argv[] ) {
    double V[3][5] = {{0,0,1,0,0}, {0,0,0,1,0}, {0,0,0,0,1}};
    double ell[5] = {0.4, 0.5, 0.6, 0.7, 0.8};
    OSBM<double,3,5>(&V, &ell);
}



