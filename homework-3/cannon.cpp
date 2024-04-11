#include <mpi.h>
#include <iostream>

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




    MPI_Finalize();
    return 0;
}