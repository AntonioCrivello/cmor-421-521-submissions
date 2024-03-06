#include <iostream>
#include <chrono>
#include <cmath>

using namespace std;
using namespace chrono;

void matmul_naive(double *A, double *B, double *C, const int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            double Cij = C[i * n + j];
            for (int k = 0; k < n; ++k)
            {
                double Aij = A[i * n + k];
                double Bjk = B[k * n + j];
                Cij += Aij * Bjk;
            }
            C[i * n + j] = Cij;
        }
    }
}

void matmul_blocked(double *A, double *B, double *C, const int n, int block_size)
{
    for (int i = 0; i < n; i += block_size)
    {
        for (int j = 0; j < n; j += block_size)
        {
            for (int k = 0; k < n; k += block_size)
            {
                // Small Matrix-Matrix Multiplication
                for (int ii = i; ii < i + block_size; ii++)
                {
                    for (int jj = j; jj < j + block_size; jj++)
                    {
                        double Cij = C[ii * n + jj];
                        for (int kk = k; kk < k + block_size; kk++)
                        {
                            Cij += A[ii * n + kk] * B[kk * n + jj];
                        }
                        C[ii * n + jj] = Cij;
                    }
                }
            }
        }
    }
}

void matmul_recursive(double *A, double *B, double *C, int rowA, int colA, int rowB,
                      int colB, int rowC, int colC, int size, int blockSize, const int n) {

    if (size <= blockSize) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                double Cij = C[(rowC + i) * n + (colC + j)];
                for (int k = 0; k < size; ++k) {
                    double Aij = A[(rowA + i) * n + (colA + k)];
                    double Bjk = B[(rowB + k) * n + (colB + j)];
                    //C[(rowC + i) * n + (colC + j)] += A[(rowA + i) * n + (colA + k)] * B[(rowB + k) * n + (colB + j)];
                    Cij += Aij * Bjk;
                }
                C[(rowC + i) * n + (colC + j)] = Cij;
            }
        }
    } else {
        int newSize = size / 2;

        // C00 = RMM(A00, B00, n / 2) + RMM(A01, B10, n / 2)
        matmul_recursive(A, B, C, rowA, colA, rowB, colB, rowC, colC, newSize, blockSize, n);
        matmul_recursive(A, B, C, rowA, colA + newSize, rowB + newSize, colB, rowC, colC, newSize, blockSize, n);

        // C01 = RMM(A00, B01, n / 2) + RMM(A01, B11, n / 2)
        matmul_recursive(A, B, C, rowA, colA, rowB, colB + newSize, rowC, colC + newSize, newSize, blockSize, n);
        matmul_recursive(A, B, C, rowA, colA + newSize, rowB + newSize, colB + newSize, rowC, colC + newSize, newSize, blockSize, n);

        // C10 = RMM(A10, B00, n / 2) + RMM(A11, B10, n / 2)
        matmul_recursive(A, B, C, rowA + newSize, colA, rowB, colB, rowC + newSize, colC, newSize, blockSize, n);
        matmul_recursive(A, B, C, rowA + newSize, colA + newSize, rowB + newSize, colB, rowC + newSize, colC, newSize, blockSize, n);

        // C11 = RMM(A10, B01, n / 2) + RMM(A11, B11, n / 2)
        matmul_recursive(A, B, C, rowA + newSize, colA, rowB, colB + newSize, rowC + newSize, colC + newSize, newSize, blockSize, n);
        matmul_recursive(A, B, C, rowA + newSize, colA + newSize, rowB + newSize, colB + newSize, rowC + newSize, colC + newSize, newSize, blockSize, n);
    }
}

int main(int argc, char *argv[])
{

    if (argc < 3)
    {
        cout << "Missing inputs." << endl;
        // Exit the programs
        exit(EXIT_FAILURE);
    }

    // User input of size for square matrix, with number of rows same as columns
    int m = atoi(argv[1]);
    int n = m;
    cout << "m = " << m << endl;
    // cout << "Matrix size: m x n = " << m << " x " << n << endl;

    // User defined block size
    int blockSize = atoi(argv[2]);

    // Allocate memory for arrays containing matrices
    double *A = new double[m * n];
    double *B = new double[m * n];
    double *C = new double[m * n];

    // Define matrices A and B as the identity matrix
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (i == j)
            {
                A[i * n + j] = 1.0;
                B[i * n + j] = 1.0;
            }
            else
            {
                A[i * n + j] = 0.0;
                B[i * n + j] = 0.0;
            }
        }
    }

    // Define matrix C as the zero matric
    for (int i = 0; i < m * n; ++i)
    {
        C[i] = 0.0;
    }
    
    // matmul_recursive(A, B, C, 0, 0, 0, 0, 0, 0, m, blockSize, n);
    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         cout << C[i + j * n] << " ";
    //     }
    //     cout << endl;
    // }

    // Number of trials to get consistent timings
    int numTrials = 50;

    // Timing for Naive Matrix-Matrix Multiplication
    high_resolution_clock::time_point startNaive = high_resolution_clock::now();
    for (int trial = 0; trial < numTrials; ++trial)
    {
        // double* copyC = new double[m * n];
        // for (int i = 0; i < m * n; ++i) {
        //     copyC[i] = 0.0;
        // }
        matmul_naive(A, B, C, m);
        // delete[] copyC;
    }
    high_resolution_clock::time_point endNaive = high_resolution_clock::now();
    duration<double> elapsedNaive = duration_cast<duration<double>>((endNaive - startNaive) / numTrials);
    //duration<double> elapsedNaive = (endNaive - startNaive) / numTrials;

    // Timing for Blocked Matrix-Matrix Multiplication
    high_resolution_clock::time_point startBlocked = high_resolution_clock::now();
    for (int trial = 0; trial < numTrials; ++trial)
    {
        // double* copyC = new double[m * n];
        // for (int i = 0; i < m * n; ++i) {
        //     copyC[i] = 0.0;
        // }
        matmul_blocked(A, B, C, m, blockSize);
        // delete[] copyC;
    }
    high_resolution_clock::time_point endBlocked = high_resolution_clock::now();
    duration<double> elapsedBlocked = duration_cast<duration<double>>((endBlocked - startBlocked) / numTrials);
    //duration<double> elapsedBlocked = (endBlocked - startBlocked) / numTrials;

    //Timing for Recursive Matrix-Matrix Multiplication
    high_resolution_clock::time_point startRecursive = high_resolution_clock::now();
    for (int trial = 0; trial < numTrials; ++trial) {
        matmul_recursive(A, B, C, 0, 0, 0, 0, 0, 0, m, blockSize, n);
    }
    high_resolution_clock::time_point endRecursive = high_resolution_clock::now();
    duration<double> elapsedRecursive = duration_cast<duration<double>>((endRecursive - startRecursive) / numTrials);
    // duration<double> elapsedRecursive = (endRecursive - startRecursive) / numTrials;


    cout << "Average Elapsed Time for Naive Implementation (seconds) = " << elapsedNaive.count() << endl;
    cout << "Average Elapsed Time for Blocked Implementation (seconds) = " << elapsedBlocked.count() << endl;
    cout << "Average Elapsed Time for Recursive Implementation (seconds) = " << elapsedRecursive.count() << endl;

    // Free allocated memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}