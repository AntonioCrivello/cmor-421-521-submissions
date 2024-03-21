#include <omp.h>
#include <iostream>

using namespace std;

#define NUM_THREADS 16
void matmulBlocked(double *A, double *B, double *C, const int n, int blockSize)
{
    for (int i = 0; i < n; i += blockSize)
    {
        for (int j = 0; j < n; j += blockSize)
        {
            for (int k = 0; k < n; k += blockSize)
            {
                // small matmul
                for (int ii = i; ii < i + blockSize; ii++)
                {
                    for (int jj = j; jj < j + blockSize; jj++)
                    {
                        double Cij = C[jj + ii * n];
                        for (int kk = k; kk < k + blockSize; kk++)
                        {
                            Cij += A[kk + ii * n] * B[jj + kk * n]; // Aik * Bkj
                        }
                        C[jj + ii * n] = Cij;
                    }
                }
            }
        }
    }
}

void matmulParallel(double *A, double *B, double *C, const int n, int blockSize, int numThreads)
{
    omp_set_num_threads(numThreads);
    for (int i = 0; i < n; i += blockSize)
    {
        // cout << omp_get_num_threads() << endl;
        for (int j = 0; j < n; j += blockSize)
        {
            for (int k = 0; k < n; k += blockSize)
            {
                #pragma omp parallel for shared(i, j, k)
                for (int ii = 0; ii < blockSize; ii++)
                {
                    for (int jj = 0; jj < blockSize; jj++)
                    {
                        for (int kk = k; kk < k + blockSize; kk++)
                        {
                            // cout << omp_get_num_threads() << endl;
                            // #pragma omp critical
                            C[(i + ii) * n + (j + jj)] += A[(i + ii) * n + kk] * B[kk * n + (j + jj)];
                        }
                    }
                }
            }
        }
    }
}

void matmulParallelCollapse(double *A, double *B, double *C, const int n, int blockSize, int numThreads)
{
    omp_set_num_threads(numThreads);
    for (int i = 0; i < n; i += blockSize)
    {
        for (int j = 0; j < n; j += blockSize)
        {
            for (int k = 0; k < n; k += blockSize)
            {
                #pragma omp parallel for collapse(2) shared(i, j, k)
                for (int ii = 0; ii < blockSize; ii++)
                {
                    for (int jj = 0; jj < blockSize; jj++)
                    {
                        for (int kk = k; kk < k + blockSize; kk++)
                        {
                            #pragma omp critical
                            C[(i + ii) * n + (j + jj)] += A[(i + ii) * n + kk] * B[kk * n + (j + jj)];
                        }
                    }
                }

                // #pragma omp single
                // {
                //     printf("Number of threads: %d\n", omp_get_num_threads());
                // }
            }
        }
    }
}

void backSolve(double *x, double *y, double *U, const int n)
{
    // Last element of x-vector is last element of y-vector
    x[n - 1] = y[n - 1];
    for (int i = n - 2; i >= 0; --i)
    {
        // Initialize sum variable as 0.
        double sum = 0.0;
        for (int j = i + 1; j < n; ++j)
        {
            // Update sum value
            sum += U[i * n + j] * x[j];
        }
        // Update ith element of x-vector
        x[i] = y[i] - sum;
    }
}

void parallelBackSolveStatic(double *x, double *y, double *U, const int n, int numThreads)
{
    omp_set_num_threads(numThreads);
    // Last element of x-vector is last element of y-vector
    x[n - 1] = y[n - 1];
    for (int i = n - 2; i >= 0; --i)
    {
        // Initialize sum variable as 0.
        double sum = 0.0;
        // Parallel sum calculations
        #pragma omp parallel
        {
            #pragma omp single
            {
                printf("Number of threads: %d\n", omp_get_num_threads());
            }

            #pragma omp for reduction(+:sum) schedule(static)
            for (int j = i + 1; j < n; ++j)
            {
                sum += U[i * n + j] * x[j];
            }
        }
        // Update ith element of x-vector
        x[i] = y[i] - sum;
    }
}


void parallelBackSolveDynamic(double *x, double *y, double *U, const int n, int numThreads)
{
    omp_set_num_threads(numThreads);
    // Last element of x-vector is last element of y-vector
    x[n - 1] = y[n - 1];
    for (int i = n - 2; i >= 0; --i)
    {
        // Initialize sum variable as 0.
        double sum = 0.0;
        // Parallel sum calculations
        #pragma omp parallel
        {
            #pragma omp single
            {
                printf("Number of threads: %d\n", omp_get_num_threads());
            }

            #pragma omp for reduction(+:sum) schedule(dynamic)
            for (int j = i + 1; j < n; ++j)
            {
                // Update sum value
                sum += U[i * n + j] * x[j];
            }
        }
        // Update ith element of x-vector
        x[i] = y[i] - sum;
    }
}


void parallelBackSolveTest(double *x, double *y, double *U, const int n, int numThreads)
{
    omp_set_num_threads(numThreads);
    // Last element of x-vector is last element of y-vector
    // x[n - 1] = y[n - 1];
    for (int j = n - 1; j >= 0; --j)
    {
        x[j] += y[j];
        // Parallel sum calculations
        #pragma omp parallel
        {
            // #pragma omp single
            // {
            //     printf("Number of threads: %d\n", omp_get_num_threads());
            // }

            #pragma omp for schedule(dynamic)
            for (int i = 0; i < j; ++i)
            {
                // Update sum value
                x[i] -= U[i * n + j] * x[j];
                // cout << "j = " << j << endl;
                // cout << U[i * n + j] << endl;
                // cout << j << endl;
            }
        }
    }
}

int main(int argc, char *argv[])
{

    if (argc < 4)
    {
        cout << "Missing inputs." << endl;
        // Exit the programs
        exit(EXIT_FAILURE);
    }

    // User input of size for square matrix, with number of rows same as columns
    int m = atoi(argv[1]);
    int n = m;

    // User defined block size
    int blockSize = atoi(argv[2]);
    int numThreads = atoi(argv[3]);

    // Allocate memory for arrays containing matrices
    double *A = new double[m * n];
    double *B = new double[m * n];
    double *C = new double[m * n];
    double *U = new double[m * n];

    // Allocate memory for vectors used in backsolve algorithm
    double *x = new double[n];
    double *y = new double[m];

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

    // double elapsedTimeBlocked = omp_get_wtime();
    // matmulBlocked(A, B, C, n, blockSize);
    // elapsedTimeBlocked = omp_get_wtime() - elapsedTimeBlocked;

    // for (int i = 0; i < m * n; ++i)
    // {
    //     C[i] = 0.0;
    // }

    // double elapsedTimeParallel = omp_get_wtime();
    // // parallel_test(C, n);
    // matmulParallel(A, B, C, n, blockSize, numThreads);
    // elapsedTimeParallel = omp_get_wtime() - elapsedTimeParallel;

    // for (int i = 0; i < m * n; ++i)
    // {
    //     C[i] = 0.0;
    // }

    // double elapsedTimeParallelCollapse = omp_get_wtime();
    // // parallel_test(C, n);
    // matmulParallelCollapse(A, B, C, n, blockSize, numThreads);
    // elapsedTimeParallelCollapse = omp_get_wtime() - elapsedTimeParallelCollapse;

    // // for (int i = 0; i < n; ++i)
    // // {
    // //     for (int j = 0; j < n; ++j)
    // //     {
    // //         cout << C[i * n + j] << " ";
    // //     }
    // //     cout << endl;
    // // }

    double elapsedBackSolve = omp_get_wtime();

    backSolve(x, y, U, n);

    elapsedBackSolve = omp_get_wtime() - elapsedBackSolve;

    for (int i = 0; i < n; ++i)
    {
        // cout << x[i] << endl;
        x[i] = 0.0;
    }

    double elapsedBackSolveTest = omp_get_wtime();

    parallelBackSolveTest(x, y, U, n, numThreads);

    elapsedBackSolveTest = omp_get_wtime() - elapsedBackSolveTest;

    for (int i = 0; i < n; ++i)
    {
        // cout << x[i] << endl;
        x[i] = 0.0;
    }

//     double elapsedParallelBackSolveStatic = omp_get_wtime();
//     parallelBackSolveStatic(x, y, U, n, numThreads);
//     elapsedParallelBackSolveStatic = omp_get_wtime() - elapsedParallelBackSolveStatic;

//     for (int i = 0; i < n; ++i)
//     {
//         cout << x[i] << endl;
//         x[i] = 0.0;
//     }

//     double elapsedParallelBackSolveDynamic = omp_get_wtime();
//     parallelBackSolveDynamic(x, y, U, n, numThreads);
//     elapsedParallelBackSolveDynamic = omp_get_wtime() - elapsedParallelBackSolveDynamic;

//     for (int i = 0; i < n; ++i)
//     {
//         //cout << x[i] << endl;
//         x[i] = 0.0;
//     }

//     // cout << elapsedTimeBlocked << endl;
//     // cout << elapsedTimeParallel << endl;
//     // cout << elapsedTimeParallelCollapse << endl;

//     cout << elapsedTimeParallelCollapse << endl;

    cout << elapsedBackSolve << endl;
    cout << elapsedBackSolveTest << endl;
//     cout << elapsedParallelBackSolveStatic << endl;
//     cout << elapsedParallelBackSolveDynamic << endl;

//     return 0;
// }
    // Free allocated memory
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] U;

    delete[] x;
    delete[] y;

    return 0;
}
