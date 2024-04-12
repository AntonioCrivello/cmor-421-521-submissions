#include <mpi.h>
#include <iostream>

#include "matrix_operations.hpp"

using namespace std;

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

    if (rank == 0) {
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
    } else {
        // Non-root processes call of MPI_Scatter
        MPI_Scatter(nullptr, local_size, MPI_DOUBLE, A_local, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(nullptr, local_size, MPI_DOUBLE, B_local, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int row_send, row_recv;
    int col_send, col_recv;
    int row_index = rank / p;
    int col_index = rank % p;

    // Circular shift of A and B
    for (int k = 0; k < p; ++k) {
        if (col_index == 0) {
            // First processor in row needs to wrap around to last
            row_send = (row_index + 1) * p - 1;
        } else {
            // All other processors send to processor directly before
            row_send = rank - 1;
        }

        if (col_index == p - 1) {
            // Last processor in row needs to receive from first
            row_recv = row_index * p;
        } else {
            // All other processors receive from processor directly after
            row_recv = rank + 1;
        }

        if (row_index == 0) {
            // First processor in columns needs to wrap around to last
            col_send = col_index + (p - 1) * p;
        } else {
            // All other processsors need to send to processor directly above
            col_send = rank - p;
        }

        if (row_index == p - 1) {
            // Last processor in column needs to receive from first 
            col_recv = col_index;
        } else {
            // All other processors need to receive from processor directly below
            col_recv = rank + p;
        }
    }
    // row_send = rank - 1;
    // if (row_send < 0) {
    //     row_send = size - 1;
    // }
    // row_recv = (rank + 1) % size;


    for (int i = 0; i < size; ++i) {
        if (rank == i) {
            cout << "Rank:" << rank << endl;
            cout << "Row Send: " << row_send << endl;
            cout << "Row Receive: " << row_recv << endl;
            cout << "Col Send: " << col_send << endl;
            cout << "Col Receive: " << col_recv << endl;
        }
    }


    // for (int k = 0; k < p; ++k) {
    //     // Local matrix multiplication
    //     matmul_local(A_local, B_local, C_local, block_size);

    //     // Circular shift row to left
    //     int row_send, row_recv;
    //     MPI_Sendrecv(A_local, local_size, MPI_DOUBLE, row_send, 0, 
    //                  A_local, local_size, MPI_DOUBLE, row_recv, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    //     // Circular shift column upward
    //     int col_send, col_recv;
    //     MPI_Sendrecv(B_local, local_size, MPI_DOUBLE, col_send, 0, 
    //                  A_local, local_size, MPI_DOUBLE, col_recv, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // }

    MPI_Finalize();
    return 0;
}