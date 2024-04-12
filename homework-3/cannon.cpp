#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>

#include "matrix_operations.hpp"

using namespace std;

void row_shift(int &row_send, int &row_recv, int row_index, int col_index, int p, int rank)
{
    if (col_index == 0)
    {
        // First processor in row needs to wrap around to last
        row_send = (row_index + 1) * p - 1;
    }
    else
    {
        // All other processors send to processor directly before
        row_send = rank - 1;
    }

    if (col_index == p - 1)
    {
        // Last processor in row needs to receive from first
        row_recv = row_index * p;
    }
    else
    {
        // All other processors receive from processor directly after
        row_recv = rank + 1;
    }
}

void column_shift(int &col_send, int &col_recv, int row_index, int col_index, int p, int rank)
{
    if (row_index == 0)
    {
        // First processor in columns needs to wrap around to last
        col_send = col_index + (p - 1) * p;
    }
    else
    {
        // All other processsors need to send to processor directly above
        col_send = rank - p;
    }

    if (row_index == p - 1)
    {
        // Last processor in column needs to receive from first
        col_recv = col_index;
    }
    else
    {
        // All other processors need to receive from processor directly below
        col_recv = rank + p;
    }
}

int main(int argc, char *argv[])
{
    // Initialize MPI Execution Environment
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Assuming number of processors is s = p x p
    int p = (int)sqrt(size);

    if (argc < 2)
    {
        if (rank == 0)
        {
            cout << "Missing input." << endl;
        }
        MPI_Finalize();
        // Exit the programs
        exit(EXIT_FAILURE);
    }

    // User input for dimensions of matrix
    int m = atoi(argv[1]);
    int n = m;

    // Calculate block size given assumption that m = n = p x b
    int block_size = m / p;

    // Size of local matrices on each processor
    int local_size = block_size * block_size;

    // Generate local matrices
    double *A_local = new double[local_size];
    double *B_local = new double[local_size];
    double *C_local = new double[local_size];

    // Initialize local C matrix
    for (int i = 0; i < local_size; ++i)
    {
        C_local[i] = 0.0;
    }

    double *A = nullptr;
    double *B = nullptr;
    double *C = nullptr;
    double *C_serial = nullptr;

    if (rank == 0)
    {
        // Root process initializes and scatter matrices used in matrix-matrix multiplication
        A = new double[m * n];
        B = new double[m * n];

        // // Randomly populate matrices
        populate_matrix(A, m, n);
        populate_matrix(B, m, n);

        // Convert matrices so that they can be scattered to the other processors
        double *A_scatter = convert_matrix(A, block_size, p, m, n);
        double *B_scatter = convert_matrix(B, block_size, p, m, n);

        // Scatter elements of array to each processor
        MPI_Scatter(A_scatter, local_size, MPI_DOUBLE, A_local, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(B_scatter, local_size, MPI_DOUBLE, B_local, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Free allocated memory for scatter matrices
        delete[] A_scatter;
        delete[] B_scatter;
    }
    else
    {
        // Non-root processes call of MPI_Scatter
        MPI_Scatter(nullptr, local_size, MPI_DOUBLE, A_local, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(nullptr, local_size, MPI_DOUBLE, B_local, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int row_send, row_recv;
    int col_send, col_recv;
    int row_index = rank / p;
    int col_index = rank % p;

    // Initial Matrix A Skewing
    for (int i = 0; i < row_index; ++i)
    {
        row_shift(row_send, row_recv, row_index, col_index, p, rank);
        // Skew A matrices on each processor
        MPI_Sendrecv_replace(A_local, local_size, MPI_DOUBLE, row_send, 0, row_recv, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Initial Matrix B Skewing
    for (int j = 0; j < col_index; ++j)
    {
        column_shift(col_send, col_recv, row_index, col_index, p, rank);
        // Skew B matrices on each processor
        MPI_Sendrecv_replace(B_local, local_size, MPI_DOUBLE, col_send, 0, col_recv, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Circular shift of A and B
    for (int k = 0; k < p; ++k)
    {
        // Local matrix multiplication
        matmul_local(A_local, B_local, C_local, block_size);

        // Determine send and receive locations each processor for its row
        row_shift(row_send, row_recv, row_index, col_index, p, rank);
        // Determine send and receive locations for each processor for its column
        column_shift(col_send, col_recv, row_index, col_index, p, rank);

        // Send and receive A and B matrices for each processor
        MPI_Sendrecv_replace(A_local, local_size, MPI_DOUBLE, row_send, 0, row_recv, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(B_local, local_size, MPI_DOUBLE, col_send, 0, col_recv, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Initialize matrix to be used to gather results from each processor
    double *C_gathered = nullptr;
    if (rank == 0)
    {
        C_gathered = new double[m * n];
    }

    // Gather all results on different processors to root
    MPI_Gather(C_local, local_size, MPI_DOUBLE, C_gathered, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        // Revert matrix back to correct order
        double *C = revert_matrix(C_gathered, block_size, p, m, n);

        // Initialize C matrix for serial matrix-matrix multiplication
        C_serial = new double[m * n];
        for (int i = 0; i < m * n; ++i)
        {
            C_serial[i] = 0.0;
        }

        // Serial matrix-matrix mutliplication
        matmul_serial(A, B, C_serial, n, block_size);

        // Check that serial result matches MPI result
        correctness_check(C, C_serial, m, n);

        // Free allocated memory
        delete[] A;
        delete[] B;
        delete[] C;
        delete[] C_gathered;
        delete[] C_serial;
    }

    // Free allocated memory for local matrices
    delete[] A_local;
    delete[] B_local;
    delete[] C_local;

    MPI_Finalize();
    return 0;
}