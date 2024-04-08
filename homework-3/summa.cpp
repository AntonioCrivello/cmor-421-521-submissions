#include <mpi.h>
#include <iostream>
#include <cstdlib>
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
    } else {
        cout << "Matrices Match" << endl;
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

    // Create communicator for each row and column
    MPI_Comm row_comm, col_comm;
    int row_color = rank / p;
    int col_color = rank % p;
    MPI_Comm_split(MPI_COMM_WORLD, row_color, rank, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, col_color, rank, &col_comm);

    double *A = nullptr;
    double *B = nullptr;
    double *C = nullptr;
    double *C_serial = nullptr;

    if (rank == 0)
    {
        // Root process initializes and scatter matrices used in matrix-matrix multiplication
        A = new double[m * n];
        B = new double[m * n];

        // Seed random number generator
        srand(time(NULL));
        // Generate random matrices A and B
        for (int i = 0; i < m * n; ++i)
        {
            // Generate random doubles
            A[i] = double (rand());
            B[i] = double (rand());
        }
        
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

    // Initialize matrices to be used for broadcast buffer
    double *A_recv = new double[local_size];
    double *B_recv = new double[local_size];

    for (int k = 0; k < p; ++k) {
        if (col_color == k) {
            // Initialize receive matrix as local matrix
            for (int i = 0; i < local_size; ++i) {
                A_recv[i] = A_local[i];    
            }
        }
        // Broadcast local matrix to row communicator group
        MPI_Bcast(A_recv, local_size, MPI_DOUBLE, k, row_comm);

        if (row_color == k) {
            // Initialize receive matrix as local matrix
            for (int i = 0; i < local_size; ++i) {
                B_recv[i] = B_local[i];    
            }
        }
        // Broadcast local matrix to column communicator group
        MPI_Bcast(B_recv, local_size, MPI_DOUBLE, k, col_comm);

        // Local Matrix-Matrix Multiplication
        matmul_local(A_recv, B_recv, C_local, block_size);
    }

    // Initialize matrix to be used to gather results from each processor
    double *C_gathered = nullptr;
    if (rank == 0) {
        C_gathered = new double[m * n];
    }

    // Gather all results on different processors to root
    MPI_Gather(C_local, local_size, MPI_DOUBLE, C_gathered, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Revert matrix back to correct order
        double *C = revert_matrix(C_gathered, block_size, p, m, n);

        // Initialize C matrix for serial matrix-matrix multiplication
        C_serial = new double[m * n];

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

    // Free communicators
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

    // Terminate MPI Execution Environment
    MPI_Finalize();
    return 0;
}