#include <iostream>
#include <chrono>
#include <cmath>

using namespace std;
using namespace chrono;

void matmul_naive(double* A, double* B, double* C, const int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double Cij = C[i * n + j];
            for (int k = 0; k < n; ++k) {
                double Aij = A[i * n + k];
                double Bjk = B[k * n + j];
                Cij += Aij * Bjk;
            }
            C[i * n + j] = Cij;
        }
    }
}

int main(int argc, char * argv[]) {
    
    if (argc < 3){
        cout << "Missing inputs." << endl;
        //Exit the programs
        exit(EXIT_FAILURE);
    }

    //User input of size for square matrix, with number of rows same as columns
    int m = atoi(argv[1]);
    int n = m;
    cout << "m = " << m << endl;
    // cout << "Matrix size: m x n = " << m << " x " << n << endl;

    //User defined block size
    int block_size = atoi(argv[2]);

    //Allocate memory for arrays containing matrices
    double* A = new double[m * n];
    double* B = new double[m * n];
    double* C = new double[m * n];

    //Define matrices A and B as the identity matrix
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) { 
                A[i * n + j] = 1.0;
                B[i * n + j] = 1.0;
            }
        }
    }

    //Define matrix C as the zero matric
    for (int i = 0; i < m * n; ++i) {
        C[i] = 0.0;
    }

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            cout << C[i * n + j] << " ";
        }
        cout << endl;
    }

    // for (int i = 0; i < m; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         cout << B[i * n + j] << " ";
    //     }
    //     cout << endl;
    // }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            cout << C[i * n + j] << " ";
        }
        cout << endl;
    }

    //Number of trials to get consistent timings
    int num_trials = 50;

    //Timing for Naive Matrix-Matrix Multiplication
    high_resolution_clock::time_point start_naive = high_resolution_clock::now();
    for (int i = 0; i < num_trials; ++i) {
        double* copyC = new double[m * n];
        for (int i = 0; i < m * n; ++i) {
            copyC[i] = 0.0;
        }
        matmul_naive(A, B, copyC, m);
        delete[] copyC;
    }
    high_resolution_clock::time_point end_naive = high_resolution_clock::now();
    duration<double> elapsed_naive = duration_cast<duration<double>>((end_naive - start_naive) / num_trials);

    cout << "Average Elapsed Time for Naive Matrix-Matrix Multiplication = " << elapsed_naive.count() << endl;


    // //Timing for Blocked Matrix-Matrix Multiplication
    // high_resolution_clock::time_point start_blocked = high_resolution_clock::now();
    // for (int i = 0; i < num_trials; ++i) {
    //     matmul_blocked
    // }
    // high_resolution_clock::time_point end_blocked = high_resolution_clock::now();
    // duration<seconds> elapsed_blocked = (end_blocked - start_blocked) / num_trials;

    // //Timing for Recursive Matrix-Matrix Multiplication
    // high_resolution_clock::time_point start_recursive = high_resolution_clock::now();
    // for (int i = 0; i < num_trials; ++i) { 
    //     matmul_recursive
    // }
    // high_resolution_clock::time_point end_recursive = high_resolution_clock::now();
    // duration<seconds> elapsed_recursive = (end_recursive - start_recursive) / num_trials;



    //Free allocated memory
    delete[] A;
    delete[] B;
    delete[] C;
    


    return 0;
}