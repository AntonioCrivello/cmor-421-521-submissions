#include <omp.h>
#include <iostream>
#include <cmath>

using namespace std;

#define BLOCK_SIZE 8

void matmulParallel(double *A, double *B, double *C, const int n, int numThreads)
{
    // Block Matrix-Matrix Multiplication using #omp parallel for
    // Define number of threads used in implementation
    omp_set_num_threads(numThreads);

    #pragma omp parallel for
    for (int i = 0; i < n; i += BLOCK_SIZE)
    {
        for (int j = 0; j < n; j += BLOCK_SIZE)
        {
            for (int k = 0; k < n; k += BLOCK_SIZE)
            {
                for (int ii = i; ii < i + BLOCK_SIZE; ii++)
                {
                    for (int jj = j; jj < j + BLOCK_SIZE; jj++)
                    {
                        for (int kk = k; kk < k + BLOCK_SIZE; kk++)
                        {
                            C[ii * n + jj] += A[ii * n + kk] * B[kk * n + jj];
                            // #pragma omp critical
                            // {
                            //     cout << omp_get_num_threads() << endl;
                            // }
                        }
                    }
                }
            }
        }
    }
}

void matmulParallelCollapsed(double *A, double *B, double *C, const int n, int numThreads)
{
    // Block Matrix-Matrix Multiplication using #omp parallel for collapse(2)
    // Define number of threads used in implementation
    omp_set_num_threads(numThreads);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i += BLOCK_SIZE)
    {
        for (int j = 0; j < n; j += BLOCK_SIZE)
        {
            for (int k = 0; k < n; k += BLOCK_SIZE)
            {
                for (int ii = i; ii < i + BLOCK_SIZE; ii++)
                {
                    for (int jj = j; jj < j + BLOCK_SIZE; jj++)
                    {
                        for (int kk = k; kk < k + BLOCK_SIZE; kk++)
                        {
                            C[ii * n + jj] += A[ii * n + kk] * B[kk * n + jj];
                            // #pragma omp critical
                            // {
                            //     cout << omp_get_num_threads() << endl;
                            // }
                        }
                    }
                }
            }
        }
    }
}

void parallelBackSolveStatic(double *x, double *y, double *U, const int n, int numThreads)
{
    omp_set_num_threads(numThreads);
    // Last element of x-vector is last element of y-vector
    // x[n - 1] = y[n - 1];
    for (int j = n - 1; j >= 0; --j)
    {
        x[j] += y[j];

        #pragma omp for schedule(static)
        for (int i = 0; i < j; ++i)
        {
            // Update sum value
            x[i] -= U[i * n + j] * x[j];
            // #pragma omp critical
            // {
            //     cout << omp_get_num_threads() << endl;
            // }
        }
    }
}

void parallelBackSolveDynamic(double *x, double *y, double *U, const int n, int numThreads)
{
    omp_set_num_threads(numThreads);
    // Last element of x-vector is last element of y-vector
    // x[n - 1] = y[n - 1];
    for (int j = n - 1; j >= 0; --j)
    {
        x[j] += y[j];
        // Parallel sum calculations

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < j; ++i)
        {
            // Update sum value
            x[i] -= U[i * n + j] * x[j];
            // #pragma omp critical
            // {
            //     cout << omp_get_num_threads() << endl;
            // }
        }
    }
}

void correctness_check(double *C, const int n)
{
    // Added correctness check from previous homework to confirm method
    // implementation.
    // Define I as the identity matrix
    double *I = new double[n * n];
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (i == j)
            {
                I[i * n + j] = 1.0;
            }
            else
            {
                I[i * n + j] = 0.0;
            }
        }
    }
    // Tolerance for machine precision
    float tol = 1e-15 * n;

    // Initial sum
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            sum += fabs(I[i * n + j] - C[i * n + j]);
        }
    }

    // Check correctness of implementation
    if (sum > tol)
    {
        cout << "Matrix C does not equal I to machine precision" << endl;
    }

    // Free allocated memory for identity matrix
    delete[] I;
}

void correctness_check_vector(double *x, const int n)
{
    // Added correctness check from previous homework to confirm method
    // implementation.
    // Define I as the identity matrix
    double *x_exact = new double[n];
    for (int i = 0; i < n; ++i)
    {
        x_exact[i] = 1.0;
    }
    // Tolerance for machine precision
    float tol = 1e-15 * n;

    // Initial sum
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
    {
        sum += fabs(x_exact[i] - x[i]);
    }

    // Check correctness of implementation
    if (sum > tol)
    {
        cout << "Vector x does not equal expected x-solution vector to machine precision" << endl;
    }

    // Free allocated memory for identity matrix
    delete[] x_exact;
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

    // User defined number of threads
    int numThreads = atoi(argv[2]);

    // Allocate memory for arrays containing matrices
    double *A = new double[m * n];
    double *B = new double[m * n];
    double *C = new double[m * n];
    double *U = new double[m * n];

    // Allocate memory for vectors used in backsolve algorithm
    double *x = new double[n];
    double *y = new double[m];

    // Define number of trials
    double numTrials = 5;

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

    // Define matrix C as the zero matrix
    for (int i = 0; i < m * n; ++i)
    {
        C[i] = 0.0;
    }

    // Define U as a unit upper triangular matrix
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (i <= j)
            {
                U[i * n + j] = 1.0;
            }
        }
    }

    // Initialize x as a zero vector
    for (int i = 0; i < n; ++i)
    {
        x[i] = 0.0;
    }

    // Define y as a vector of n to 1
    int val = 1.0;
    for (int j = m - 1; j >= 0; --j)
    {
        y[j] = val;
        val++;
    }

    double elapsedParallel = 0.0;
    for (int trial = 0; trial < numTrials; ++trial)
    {
        for (int i = 0; i < m * n; ++i)
        {
            C[i] = 0.0;
        }
        double elapsedParallelStart = omp_get_wtime();
        matmulParallel(A, B, C, n, numThreads);
        elapsedParallel += omp_get_wtime() - elapsedParallelStart;
        correctness_check(C, n);
    }
    elapsedParallel /= numTrials;

    double elapsedParallelCollapsed = 0.0;
    for (int trial = 0; trial < numTrials; ++trial)
    {
        for (int i = 0; i < m * n; ++i)
        {
            C[i] = 0.0;
        }
        double elapsedParallelCollapsedStart = omp_get_wtime();
        matmulParallelCollapsed(A, B, C, n, numThreads);
        elapsedParallelCollapsed += omp_get_wtime() - elapsedParallelCollapsedStart;
        correctness_check(C, n);
    }
    elapsedParallelCollapsed /= numTrials;

    double elapsedParallelBackSolveStatic = 0.0;
    for (int trial = 0; trial < numTrials; ++trial)
    {
        for (int i = 0; i < n; ++i)
        {
            x[i] = 0.0;
        }
        double elapsedParallelBackSolveStaticStart = omp_get_wtime();
        parallelBackSolveStatic(x, y, U, n, numThreads);
        elapsedParallelBackSolveStatic += omp_get_wtime() - elapsedParallelBackSolveStaticStart;
        correctness_check_vector(x, n);
    }
    elapsedParallelBackSolveStatic /= numTrials;

    double elapsedParallelBackSolveDynamic = 0.0;
    for (int trial = 0; trial < numTrials; ++trial)
    {
        for (int i = 0; i < n; ++i)
        {
            x[i] = 0.0;
        }
        double elapsedParallelBackSolveDynamicStart = omp_get_wtime();
        parallelBackSolveDynamic(x, y, U, n, numThreads);
        elapsedParallelBackSolveDynamic += omp_get_wtime() - elapsedParallelBackSolveDynamicStart;
        correctness_check_vector(x, n);
    }
    elapsedParallelBackSolveDynamic /= numTrials;

    cout << "Blocked Matrix-Matrix Multiplication with #omp parallel for: " << elapsedParallel << endl;
    cout << "Blocked Matrix-Matrix Multiplication with #omp parallel for collapse(2): " << elapsedParallelCollapsed << endl;
    cout << "Parallel Back Solve with static scheduling: " << elapsedParallelBackSolveStatic << endl;
    cout << "Parallel Back Solve with dynamic scheduling: " << elapsedParallelBackSolveDynamic << endl;

    // Free allocated memory
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] U;

    delete[] x;
    delete[] y;

    return 0;
}
