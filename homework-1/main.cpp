#include <iostream>
#include <chrono>
#include <cmath>
#include "matrix.hpp"

using namespace std;
using namespace std::chrono;

#define BLOCK_SIZE 16

void matmul_naive(Matrix * A, Matrix * B, Matrix * C, const int n){
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            for (int k = 0; k < n; ++k){
                (*C)(i, j) += (*A)(i, k) * (*B)(k, j);
            }
        }
    }
}

void matmul_blocked(Matrix * A, Matrix * B, Matrix * C, const int n, int block_size){
  for (int i = 0; i < n; i += block_size){
    for (int j = 0; j < n; j += block_size){
        for (int k = 0; k < n; k += block_size){

	    // small matmul
	    for (int ii = i; ii < i + block_size; ii++){
	        for (int jj = j; jj < j + block_size; jj++){
                double tempCij = (*C)(ii, jj);
                //(*C)(i,j) = (*C)(ii, jj);
	            for (int kk = k; kk < k + block_size; kk++){
                    tempCij += (*A)(ii, kk) * (*B)(kk, jj);
                    //(*C)(i,j) += (*A)(ii, kk) * (*B)(kk, jj);
	            }
                //(*C)(ii, jj) = (*C)(i, j);
                (*C)(ii, jj) = tempCij;
	        }
	    }
      }
    }
  }
}

void matmul_recursive(Matrix * A, Matrix * B, Matrix * C, int newSize, int size) {

    if (size <= BLOCK_SIZE) {
        for (int i = 0; i < newSize; i++){
            for (int j = 0; j < newSize; j++){
                (* C)(i,j) += (* A)(i,j) * (* B)(i,j) ;
            }
        }
 
    } else {
        
        //C00 = RMM(A00, B00, n / 2) + RMM(A01, B10, n / 2)
        matmul_recursive(A, B, C, newSize / 2, newSize);
        matmul_recursive(A + newSize, B, C, newSize / 2, newSize);

        //C01 = RMM(A00, B01, n / 2) + RMM(A01, B11, n / 2)
        matmul_recursive(A, B + newSize, C, newSize / 2, newSize);
        matmul_recursive(A + newSize, B + newSize, C, newSize / 2, newSize);

        //C10 = RMM(A10, B00, n / 2) + RMM(A11, B10, n / 2)
        matmul_recursive(A + newSize, B, C + newSize, newSize / 2, newSize);
        matmul_recursive(A + 2 * newSize, B, C + newSize, newSize / 2, newSize);

        //C11 = RMM(A10, B01, n / 2) + RMM(A11, B11, n / 2)
        matmul_recursive(A + newSize, B + newSize, C + newSize, newSize / 2, newSize);
        matmul_recursive(A + 2 * newSize, B + newSize, C + newSize, newSize / 2, newSize);

    }

}

int main(int argc, char * argv[]) {

    if (argc < 2){

        cout << "Missing inputs." << endl;
        //Exit the program
        exit(EXIT_FAILURE);

    }

    //User input of number of rows
    int m = atoi(argv[1]);
    //Given matrix is square, number of rows are equal to number of columns
    int n = m;
    cout << "Matrix size: m x n = " << m << " x " << n << endl;

    //Size of matrices A, B, and C
    int matrixSize = n;
    int blockSize = 16;

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

    int numTrials = 1000;

    //Timings for Naive Matrix-Matrix Multiplcation
    high_resolution_clock::time_point startNaive = high_resolution_clock::now();

    for (int i = 0; i < numTrials; i++){
        Matrix copyC = C;
        matmul_naive(& A, & B, & copyC, matrixSize);
            
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
    
    for (int i = 0; i < numTrials; i++){
        Matrix copyC = C;
        matmul_blocked(& A, & B, & copyC, matrixSize, blockSize);

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

    // //Timings for Recursive Matrix-Matrix Multiplication
    // high_resolution_clock::time_point startRecursive = high_resolution_clock::now();
    
    // for (int i = 0; i < numTrials; i++){
    //     Matrix copyC = C;
    //     matmul_recursive(& A, & B, & C, matrixSize / 2, matrixSize);

    //     cout << "\n" "" << endl;
    //     // Print matrix C
    //     for (int i = 0; i < m; i++) {
    //         for (int j = 0; j < n; j++) {
    //             cout << copyC(i, j) << " ";
    //         }
    //         cout << endl;
    //     }
    // }

    // high_resolution_clock::time_point endRecursive = high_resolution_clock::now();
    // duration<double> elapsed_matmul_recursive = (endRecursive - startRecursive) / numTrials;

    cout << "Elapsed time for naive matrix-matrix multiplication  = " << elapsed_matmul_naive.count() << endl; 
    cout << "Elapsed time for blocked matrix-matrix multiplication  = " << elapsed_matmul_blocked.count() << endl; 
    //cout << "Elapsed time for recursive matrix-matrix multiplication  = " << elapsed_matmul_recursive.count() << endl; 

    return 0;

    //Had to do this
    //chmod +x generate-timings.sh
}

