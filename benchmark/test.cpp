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

template<typename T, size_t rows, size_t cols>
void print_mat(double M[rows][cols]);

template <typename T, size_t rows, size_t cols>
void printval(T (*array)[rows][cols]){
    int i,j;
    T rownorms[rows];
    T rowtemp[cols];
    /**array[0][0] = 1.0;
    std::cout << *array[0][1] << '\n';*/
    /*for (i=0; i<rows; i++) {
        for (j=0; j<cols; j++) {
            rowtemp[j] = (*array)[i][j];
            std::cout << (*array)[i][j] << '\n';
        }
        rownorms[i] = blas::dot(cols,rowtemp,1,rowtemp,1);
    }

    for (i=0;i<rows;i++) {
        std::cout << rownorms[i] << '\n';
    }*/
    /*double* critrow = (*array)[2];*/
    std::cout << array[0][1] << "\n";
}

template<typename T, size_t len>
void rot(T (*array)[len]){
    blas::rot(4, array[1], 1, array[2], 1, 0.1, 0.8);
}


int main( int argc, char *argv[] ) {
    int i;

    /*double V[3][4] = {{0,1,2,3}, {4,5,6,7}, {8,9,10,11}};*/
    double V[12] = {0,1,2,3,4,5,6,7,8,9,10,11};
    /*rot<double,12>(&V);*/
    /*for (i=0; i<10;i++){
        std::cout << V[i] << "\n";
    }*/
    /*std::cout << V[0][0];*/
    /*double test[4] = {0,1,2,3};
    double test2[4] = {8,9,10,11};
    blas::rot(4,(&test)[0],1,(&test2)[0],1,0.1,0.8);
    std::cout << test2[3];*/
    blas::rot(4,(&V)[0],1,(&V)[5],1,0.1,0.8);
    std::cout << V[0];


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


