#include "matrix_operations.hpp"

#include <mpi.h>
#include <iostream>
#include <cmath>
#include <ctime>

using namespace std;

double *convert_matrix(double *matrix, int block_size, int p, int m, int n)
{
    // Reorders elements in provided matrix so that each processor receives correct elements
    // to store.

    // Create temporary matrix
    double *temp = new double[m * n];

    // Index for temporary matrix
    int index = 0;

    for (int i_b = 0; i_b < p; ++i_b)
    {
        for (int j_b = 0; j_b < p; ++j_b)
        {
            for (int i = 0; i < block_size; ++i)
            {
                for (int j = 0; j < block_size; ++j)
                {
                    temp[index++] = matrix[(i + i_b * block_size) * n + (j + j_b * block_size)];
                }
            }
        }
    }
    return temp;
}

double *revert_matrix(double *matrix, int block_size, int p, int m, int n)
{
    // Reorders elements in provided matrix so that root processor result is not scattered.

    // Create temporary matrix
    double *temp = new double[m * n];

    // Index for scattered matrix
    int index = 0;

    for (int i_b = 0; i_b < p; ++i_b)
    {
        for (int j_b = 0; j_b < p; ++j_b)
        {
            for (int i = 0; i < block_size; ++i)
            {
                for (int j = 0; j < block_size; ++j)
                {
                    temp[(i + i_b * block_size) * n + (j + j_b * block_size)] = matrix[index++];
                }
            }
        }
    }
    return temp;
}

void matmul_local(double *A_local, double *B_local, double *C_local, int block_size)
{
    for (int i = 0; i < block_size; ++i)
    {
        for (int j = 0; j < block_size; ++j)
        {
            double Cij = C_local[i * block_size + j];
            for (int k = 0; k < block_size; ++k)
            {
                Cij += A_local[i * block_size + k] * B_local[k * block_size + j];
            }
            C_local[i * block_size + j] = Cij;
        }
    }
}

void matmul_serial(double *A, double *B, double *C, const int n, int block_size)
{
    // Blocked Matrix-Matrix Multiplication
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

void populate_matrix(double *matrix, int m, int n){
    srand(time(NULL));
    for (int i = 0; i < m * n; ++i) {
        matrix[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * RAND_MAX;
    }
}

void correctness_check(double *C, double *C_serial, int m, int n)
{
    // Tolerance for machine precision
    float tol = 1e-15 * n;
    // Initial sum
    double sum = 0.0;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            sum += fabs(C_serial[i * n + j] - C[i * n + j]);
        }
    }

    // Check correctness of implementation
    if (sum > tol)
    {
        cout << "Matrix C does not equal C from serial routine to machine precision" << endl;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                cout << C[i * n + j] << " ";
            }
            cout << endl;
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                cout << C_serial[i * n + j] << " ";
            }
            cout << endl;
        }
    }
    else
    {
        cout << "Matrices Match" << endl;
    }
}