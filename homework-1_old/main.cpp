#include <iostream>
#include <chrono>
#include <cmath>
#include "matrix.hpp"

using namespace std;
using namespace std::chrono;

//#define BLOCK_SIZE 64

void matmul_naive(Matrix* A, Matrix* B, Matrix* C, const int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                (*C)(i, j) += (*A)(i, k) * (*B)(k, j);
            }
        }
    }
}


void matmul_blocked(Matrix* A, Matrix* B, Matrix* C, const int n, int blockSize) {
  for (int i = 0; i < n; i += blockSize) {
    for (int j = 0; j < n; j += blockSize) {
        for (int k = 0; k < n; k += blockSize) {
	    // small matmul
	        for (int ii = i; ii < i + blockSize; ii++) {
	            for (int jj = j; jj < j + blockSize; jj++) {
                    double tempCij = (*C)(ii, jj);
	                for (int kk = k; kk < k + blockSize; kk++) {
                        tempCij += (*A)(ii, kk) * (*B)(kk, jj);
	                }
                    (*C)(ii, jj) = tempCij;
	            }   
	        }
        } 
    }
  }
}


void matmul_recursive(Matrix* A, Matrix* B, Matrix* C, int rowA, int colA, int rowB, int colB, int rowC, int colC, int size, int blockSize){
    if (size <= blockSize) {
         for (int i = 0; i < size; ++i) {
            for (int k = 0; k < size; ++k) {
                for (int j = 0; j < size; ++j) {
                    (*C)(rowC + i, colC + j) += (*A)(rowA + i, colA + k) * (*B)(rowB + k, colB + j);
                }
            }
        }
    } else {
        int newSize = size / 2;
        //C00 = RMM(A00, B00, n / 2) + RMM(A01, B10, n / 2)
        matmul_recursive(A, B, C, rowA, colA, rowB, colB, rowC, colC, newSize, blockSize);
        matmul_recursive(A, B, C, rowA, colA + newSize, rowB + newSize, colB, rowC, colC, newSize, blockSize);

        //C01 = RMM(A00, B01, n / 2) + RMM(A01, B11, n / 2)
        matmul_recursive(A, B, C, rowA, colA, rowB, colB + newSize, rowC, colC + newSize, newSize, blockSize);
        matmul_recursive(A, B, C, rowA, colA + newSize, rowB + newSize, colB + newSize, rowC, colC + newSize, newSize, blockSize);

        //C10 = RMM(A10, B00, n / 2) + RMM(A11, B10, n / 2)
        matmul_recursive(A, B, C, rowA + newSize, colA, rowB, colB, rowC + newSize, colC, newSize, blockSize);
        matmul_recursive(A, B, C, rowA + newSize, colA + newSize, rowB + newSize, colB, rowC + newSize, colC, newSize, blockSize);

        //C11 = RMM(A10, B01, n / 2) + RMM(A11, B11, n / 2)
        matmul_recursive(A, B, C, rowA + newSize, colA, rowB, colB + newSize, rowC + newSize, colC + newSize, newSize, blockSize);
        matmul_recursive(A, B, C, rowA + newSize, colA + newSize, rowB + newSize, colB + newSize, rowC + newSize, colC + newSize, newSize, blockSize);
    }
}


int main(int argc, char * argv[]) {
    if (argc < 3){
        cout << "Missing inputs." << endl;
        //Exit the program
        exit(EXIT_FAILURE);
    }

    //User input of number of rows
    int m = atoi(argv[1]);
    //Given matrix is square, number of rows are equal to number of columns
    int n = m;
    cout << "Matrix size: m x n = " << m << " x " << n << endl;

    //User input of block size
    int blockSize = atoi(argv[2]);

    //Size of matrices A, B, and C
    int matrixSize = n;

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


    // // Print matrix C
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++) {
    //         cout << C(i, j) << " ";
    //     }
    //     cout << endl;
    // }

    int numTrials = 50;

    //Timings for Naive Matrix-Matrix Multiplcation
    high_resolution_clock::time_point startNaive = high_resolution_clock::now();

    for (int i = 0; i < numTrials; i++) {
        Matrix copyC = C;
        matmul_naive(&A, &B, &copyC, matrixSize);
            
        // cout << "\n" "" << endl;
        // // Print matrix C
        // for (int i = 0; i < m; i++) {
        //     for (int j = 0; j < n; j++) {
        //         cout << copyC(i, j) << " ";
        //     }
        //     cout << endl;
        // }
        //cout << i << endl;

    }

    //cout << "=============================================================" << endl; 

    high_resolution_clock::time_point endNaive = high_resolution_clock::now();
    duration<double> elapsed_matmul_naive = (endNaive - startNaive) / numTrials;

    //Timings for Blocked Matrix-Matrix Multiplication
    high_resolution_clock::time_point startBlocked = high_resolution_clock::now();
    
    for (int i = 0; i < numTrials; i++) {
        Matrix copyC = C;
        matmul_blocked(&A, &B, &copyC, matrixSize, blockSize);

        // cout << "\n" "" << endl;
        // // Print matrix C
        // for (int i = 0; i < m; i++) {
        //     for (int j = 0; j < n; j++) {
        //         cout << copyC(i, j) << " ";
        //     }
        //     cout << endl;
        // }

        //cout << i << endl;

    }

    high_resolution_clock::time_point endBlocked = high_resolution_clock::now();
    duration<double> elapsed_matmul_blocked = (endBlocked - startBlocked) / numTrials;

    //Timings for Recursive Matrix-Matrix Multiplication
    high_resolution_clock::time_point startRecursive = high_resolution_clock::now();
    
    for (int k = 0; k < numTrials; k++) {
        Matrix copyC = C;
        matmul_recursive(&A, &B, &copyC, 0, 0, 0, 0, 0, 0, matrixSize, blockSize);

        // if (k == 0){
        //     cout << "\n" "" << endl;
        //     // Print matrix C
        //     for (int i = 0; i < m; i++) {
        //         for (int j = 0; j < n; j++) {
        //             cout << copyC(i, j) << " ";
        //         }
        //         cout << endl;
        //     }
        // }

    }

    high_resolution_clock::time_point endRecursive = high_resolution_clock::now();
    duration<double> elapsed_matmul_recursive = (endRecursive - startRecursive) / numTrials;

    cout << "Elapsed time for naive matrix-matrix multiplication  = " << elapsed_matmul_naive.count() << endl; 
    cout << "Elapsed time for blocked matrix-matrix multiplication  = " << elapsed_matmul_blocked.count() << endl; 
    cout << "Elapsed time for recursive matrix-matrix multiplication  = " << elapsed_matmul_recursive.count() << endl; 

    return 0;

}

