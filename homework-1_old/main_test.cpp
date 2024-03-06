#include <iostream>
#include <chrono>
#include <cmath>
// #include "matrix.hpp"

using namespace std;
using namespace std::chrono;

//#define BLOCK_SIZE 64

void matmul_naive(double* A, double* B, double* C, const int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

void matmul_blocked(double* A, double* B, double* C, const int n, int blockSize) {
  for (int i = 0; i < n; i += blockSize) {
    for (int j = 0; j < n; j += blockSize) {
        for (int k = 0; k < n; k += blockSize) {
	    // small matmul
	        for (int ii = i; ii < i + blockSize; ii++) {
	            for (int jj = j; jj < j + blockSize; jj++) {
                    double Cij = C[ii * n + jj];
	                for (int kk = k; kk < k + blockSize; kk++) {
                        Cij += A[ii * n + kk] * B[kk * n + jj];
	                }
                    C[ii * n + jj] = Cij;
	            }   
	        }
        } 
    }
  }
}

void matmul_recursive(double * A, double * B, double * C, int rowA, int colA, int rowB, int colB, int rowC, int colC, int size, int blockSize){
    if (size <= blockSize) {
         for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                for (int k = 0; k < size; ++k) {
                    C[(rowC + i) * size + (colC + j)] += A[(rowA + i) * size + (colA + k)] * B[(rowB + k) * size + (colB + j)];
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

    //User input of block size
    int blockSize = atoi(argv[2]);

    cout << "Matrix size n = " << m << " x " << n << " , Block Size = " << blockSize << endl;

    //Size of matrices A, B, and C
    int matrixSize = n;

    //Define Matrices
    double * A = new double [m * n];
    double * B = new double [m * n];
    double * C = new double [m * n];

    // A[0] = 1;
    // A[0 + 1] = -1;
    // for (int i = 1; i < m - 1; ++i){
    //     A[i * m + i - 1] = -1;
    //     A[i * m + i] = 2;
    //     A[i * m + i + 1] = -1;
    // }
    // A[(m - 1) * m + n - 2] = -1;
    // A[(m - 1) * m + n - 1] = 1;

    // Initialize A and B as identity matrices
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                A[i * n + j] = 1.0;
                B[i * n + j] = 1.0;
            } else {
                A[i * n + j] = 0.0;
                B[i * n + j] = 0.0;
            }
        }
    }

    //Initialize C as zero matrix
    for (int i = 0; i < m * n; i++) {
        C[i] = 0.0;
    }

    int numTrials = 1;

    cout << "=============================================================" << endl; 
    
    //Timings for Naive Matrix-Matrix Multiplcation
    high_resolution_clock::time_point startNaive = high_resolution_clock::now();

    for (int i = 0; i < numTrials; i++) {
        // Reset matrix C to zero before each trial
        for (int i = 0; i < m * n; i++) {
            C[i] = 0.0;
        }
        matmul_naive(A, B, C, matrixSize);
            
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                cout << C[i * n + j] << " ";
            }
            cout << endl;
        }

    }

    high_resolution_clock::time_point endNaive = high_resolution_clock::now();
    duration<double> elapsed_matmul_naive = (endNaive - startNaive) / numTrials;

    cout << "=============================================================" << endl; 

    //Timings for Blocked Matrix-Matrix Multiplication
    high_resolution_clock::time_point startBlocked = high_resolution_clock::now();

    for (int i = 0; i < numTrials; i++) {
        for (int i = 0; i < m * n; i++) {
            C[i] = 0.0;
        }
        matmul_blocked(A, B, C, matrixSize, blockSize);
            
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                cout << C[i * n + j] << " ";
            }
            cout << endl;
        }

    }

    high_resolution_clock::time_point endBlocked = high_resolution_clock::now();
    duration<double> elapsed_matmul_blocked = (endBlocked - startBlocked) / numTrials;

    cout << "=============================================================" << endl; 

    //Timings for Recursive Matrix-Matrix Multiplication
    high_resolution_clock::time_point startRecursive = high_resolution_clock::now();
    
    for (int i = 0; i < numTrials; i++) {
        // Reset matrix C to zero before each trial
        for (int i = 0; i < m * n; i++) {
            C[i] = 0.0;
        }
        matmul_recursive(A, B, C, 0, 0, 0, 0, 0, 0, matrixSize, blockSize);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                cout << C[i * n + j] << " ";
            }
            cout << endl;
        }

    }

    high_resolution_clock::time_point endRecursive = high_resolution_clock::now();
    duration<double> elapsed_matmul_recursive = (endRecursive - startRecursive) / numTrials;

    cout << "Elapsed time for naive matrix-matrix multiplication  = " << elapsed_matmul_naive.count() << endl; 
    cout << "Elapsed time for blocked matrix-matrix multiplication  = " << elapsed_matmul_blocked.count() << endl; 
    cout << "Elapsed time for recursive matrix-matrix multiplication  = " << elapsed_matmul_recursive.count() << endl; 

    //Freeing allocated memory
    delete [] A;
    delete [] B;
    delete [] C;

    return 0;

}

