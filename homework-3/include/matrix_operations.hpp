// Common matrix operations needed for boh SUMMMA and Canon's algorithm
#ifndef MATRIX_OPERATIONS
#define MATRIX_OPERATIONS

#include <mpi.h>

// Functions used in matrix scatter and gather
double *convert_matrix(double *matrix, int block_size, int p, int m, int n);
double *revert_matrix(double *matrix, int block_size, int p, int m, int n);

// Functions used in matrix-matrix multiplication
void matmul_local(double *A_local, double *B_local, double *C_local, int block_size);
void matmul_serial(double *A, double *B, double *C, const int n, int block_size);

// Function used to randomly generate matrices
void populate_matrix(double *matrix, int m, int n);

// Function used in checking algorithm implementation
void correctness_check(double *C, double *C_serial, int m, int n);

#endif
