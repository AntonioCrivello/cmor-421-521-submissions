#include <mpi.h>
#include <iostream>

using namespace std;

double *convert_matrix(double *matrix, int block_size, int p, int m, int n) {
    // Reorders elements in provided matrix so that each processor receives correct elements
    // to store.

    // Create temporary matrix
    double *temp = new double [m * n];

    // Index for temporary matrix
    int index = 0;

    for (int i_b = 0; i_b < p; ++i_b) {
        for (int j_b = 0; j_b < p; ++j_b) {
            for (int i = 0; i < block_size; ++i) {
                for (int j = 0; j < block_size; ++j) {
                    temp[index++] = matrix[(i + i_b * block_size) * n + (j + j_b * block_size)];
                }
            }
        }
    }

    return temp;
}


int main(int argc, char *argv[]) {

    if (argc < 2) {
        cout << "Missing input." << endl;
        // Exit the programs
        exit(EXIT_FAILURE);
    }

    // User input for dimensions of matrix
    int m = atoi(argv[1]);
    int n = m;

    // Initialize MPI Execution Environment
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Assuming number of processors is s = p x p
    int p = (int) sqrt(size);

    // Calculate block size given assumption that m = n = p x b
    int block_size = m / p;

    // Create communicator for each row and column
    int row_color = rank / p;
    int col_color = rank % p;

    // Split the communicator based on the row color
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row_color, rank, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, col_color, rank, &col_comm);

    if (rank == 0) {

        double *A = new double [m * n];  
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j) {
                    A[i * n + j] = 1.0;
                } else if (i <= j) {
                    A[i * n + j] = 0.0;
                } else {
                    A[i * n + j] = 2.0;
                }
            }
        }

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                cout << A[i * n + j] << " ";
            }
            cout << " " << endl;
        }

    // if (rank == 0) {
        cout << "Number of processors: " << p * p << endl;
        cout << "Block Size " << block_size << endl;
    // }

    // if (rank == 0) {
        double *temp = convert_matrix(A, block_size, p, m, n);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                cout << temp[i * n + j] << " ";
            }
            cout << " " << endl;
        }
    }

    // Terminate MPI Execution Environment 
    MPI_Finalize();









    // int m = 16;
    // int n = 16;

    // // Allocate memory for arrays containing matrices
    // double *A = new double[m * n];
    // double *B = new double[m * n];
    // double *C = new double[m * n];

    // for (int i = 0; i < m; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         if (i == j) {
    //             A[i * n + j] = 1.0;
    //         } else if (i <= j) {
    //             A[i * n + j] = 0.0;
    //         } else {
    //             A[i * n + j] = 2.0;
    //         }
    //     }
    // }

    // for (int i = 0; i < m; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         cout << A[i * n + j] << " ";
    //     }
    //     cout << " " << endl;
    // }

    // double * matrix_A = flatten_array(A, 4, m, n);


    // for (int i = 0; i < m; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         cout << matrix_A[i * n + j] << " ";
    //     }
    //     cout << " " << endl;
    // }
    

    // // Initialization of MPI Environment
    // MPI_Init(NULL, NULL);

    // int rank, size;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Comm_size(MPI_COMM_WORLD, &size);

    // // Assuming size is p x p
    // int p = (int) sqrt(size);

    // // Block size each processor has
    // int block_size = m * n / p;

    // // Create communicator for each row and column
    // int row_color = rank / p;
    // int col_color = rank % p;

    // // Split the communicator based on the row color
    // MPI_Comm row_comm, col_comm;
    // MPI_Comm_split(MPI_COMM_WORLD, row_color, rank, &row_comm);
    // MPI_Comm_split(MPI_COMM_WORLD, col_color, rank, &col_comm);


    // double * A_local = new double [b * b];
    // double * A_recv = new double [b * b];
    // // // Split the communicator based on the column color
    // // MPI_Comm col_comm;
    // // MPI_Comm_split(MPI_COMM_WORLD, col_color, rank, &col_comm);

    // double* A_scatter = new double [m * n];
    // double* B_scatter = new double [m * n];




    // if (rank == 0) {
    //     // Scatter chunks of the arrays to the other ranks
    //     MPI_Scatter(A_scatter, block_size, MPI_DOUBLE, )
        
    // }

    // int x = rank;
    // for (int k = 0; k < p; ++k) {

    //     MPI_Bcast(A_recv, b * b, MPI_DOUBLE, k, row_comm);


    //     // Broadcast along the row
    //     MPI_Bcast(&k, 1, MPI_FLOAT, 0, row_comm);
    
    //     // Broadcast along the column
    //     MPI_Bcast(&k, 1, MPI_FLOAT, 0, col_comm);

    //     // Gather the results from the row
    //     MPI_Gather()

    //     // Gather the results from the column
    //     MPI_Gather()

    // }

    // // cout << "On rank " << rank;
    // // cout << "(Row, Col) = " << row_color << ", " << col_color << endl;

    // // int x = rank;
    // // MPI_Bcast(&x, 1, MPI_INT, 0, row_comm);
    
    // // cout << "After bcast, on rank " << rank << ", x = " << x << endl;


    // // Free communicator objects
    // MPI_Comm_free(&row_comm);
    // MPI_Comm_free(&col_comm);

    // // Terminate MPI Execution Environment 
    // MPI_Finalize();
    
    // Free memory allocated for arrays
    // delete[] A;
    // delete[] B;
    // delete[] C;

    return 0; 
}