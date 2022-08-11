#include <blas.hh>
#include <lapack.hh>
#include <RandBLAS.hh>

#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <chrono>

template <typename T, size_t rows, size_t cols>
void printval(T (*array)[rows][cols]){
    int i,j;
    T rownorms[rows];
    T rowtemp[cols];
    /**array[0][0] = 1.0;
    std::cout << *array[0][1] << '\n';*/
    for (i=0; i<rows; i++) {
        for (j=0; j<cols; j++) {
            rowtemp[j] = (*array)[i][j];
            /*std::cout << (*array)[i][j] << '\n';*/
        }
        rownorms[i] = blas::dot(cols,rowtemp,1,rowtemp,1);
    }

    for (i=0;i<rows;i++) {
        std::cout << rownorms[i] << '\n';
    }
}


int main( int argc, char *argv[] ) {

    double V[3][4] = {{0,1,2,3}, {4,5,6,7}, {8,9,10,11}};
    printval<double,3,4>(&V);
    /*std::cout << V[0][0];*/

}
