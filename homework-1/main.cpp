#include <iostream>
#include <chrono>
#include <cmath>
#include "matrix.hpp"

using namespace std;
using namespace std::chrono;

#define BLOCK_SIZE 16

void recursive_matmul(Matrix * A, Matrix * B, Matrix * C, int newSize, int size) {

    if (size <= BLOCK_SIZE) {
        for (int i = 0; i < size; i++){
            for (int j = 0; j < size; j++){
                (* C)(i,j) += (* A)(i,j) * (* B)(i,j) ;
            }
        }
 
    } else {
        
        //C00 = RMM(A00, B00, n / 2) + RMM(A01, B10, n / 2)
        recursive_matmul(A, B, C, newSize / 2, newSize);
        recursive_matmul(A + newSize, B, C, newSize / 2, newSize);

        //C01 = RMM(A00, B01, n / 2) + RMM(A01, B11, n / 2)
        recursive_matmul(A, B + newSize, C, newSize / 2, newSize);
        recursive_matmul(A + newSize, B + newSize, C, newSize / 2, newSize);

        //C10 = RMM(A10, B00, n / 2) + RMM(A11, B10, n / 2)
        recursive_matmul(A + newSize, B, C + newSize, newSize / 2, newSize);
        recursive_matmul(A + 2 * newSize, B, C + newSize, newSize / 2, newSize);

        //C11 = RMM(A10, B01, n / 2) + RMM(A11, B11, n / 2)
        recursive_matmul(A + newSize, B + newSize, C + newSize, newSize / 2, newSize);
        recursive_matmul(A + 2 * newSize, B + newSize, C + newSize, newSize / 2, newSize);

    }


}




//Compile command:
//g++ -c src/matrix.cpp -o main -I./include -O3 -std=c++11 

int main(void) {

    int m = 4;
    int n = 4;

    //Define Matrices
    Matrix A = Matrix(m, n);
    Matrix B = Matrix(m, n);
    Matrix C = Matrix(m, n);

    // Initialize matrix A and B as identity matrices
    // Initialize matrix C as zero matrix
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {

                A(i, j) = 1.0;
                B(i, j) = 1.0;

            } else {

                A(i, j) = 0.0;
                B(i, j) = 0.0;

            }

            C(i, j) = 0.0;

        }
    }


    // Print matrix C
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << A(i, j) << " ";
        }
        std::cout << std::endl;
    }

    return 0;

}

