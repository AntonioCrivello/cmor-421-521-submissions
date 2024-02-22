#include <iostream>
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono;

//Compilation Command:
//g++ -O3 -o matmul matmul.cpp
//Run Command:
//./matmul matrix_size block_size

// computes C = C + A*B
void matmul_naive(const int n, double* C, double* A, double* B){
  for (int i = 0; i < n; ++i){
    for (int j = 0; j < n; ++j){
      double Cij = C[j + i * n];
      for (int k = 0; k < n; ++k){
	double Aij = A[k + i * n];
	double Bjk = B[j + k * n];	
	Cij += Aij * Bjk;
      }
      C[j + i * n] = Cij;
    }
  }
}

void matmul_blocked(const int n, double* C, double* A, double* B, int block_size){
  for (int i = 0; i < n; i += block_size){
    for (int j = 0; j < n; j += block_size){
      for (int k = 0; k < n; k += block_size){

	// small matmul
	for (int ii = i; ii < i + block_size; ii++){
	  for (int jj = j; jj < j + block_size; jj++){
	    double Cij = C[jj + ii * n];
	    for (int kk = k; kk < k + block_size; kk++){
	      Cij += A[kk + ii * n] * B[jj + kk * n]; // Aik * Bkj
	    }
	    C[jj + ii * n] = Cij;
	  }
	}
	
      }
    }
  }
}

int main(int argc, char * argv[]){

  int n = atoi(argv[1]);
  //Makes block_size an input
  int block_size = atoi(argv[2]);

  cout << "Matrix size n = " << n << ", block size = " << block_size << endl;
  
  double * A = new double[n * n];
  double * B = new double[n * n];
  double * C = new double[n * n];

  // make A, B = I
  for (int i = 0; i < n; ++i){
    A[i + i * n] = 1.0;
    B[i + i * n] = 1.0;
  }
  for (int i = 0; i < n * n; ++i){
    C[i] = 0.0;
  }

  int num_trials = 100;

  // Measure performance
  high_resolution_clock::time_point start = high_resolution_clock::now();
  for (int i = 0; i < num_trials; ++i){
    matmul_naive(n, C, A, B);
  }
  high_resolution_clock::time_point end = high_resolution_clock::now();
  duration<double> elapsed_naive = (end - start) / num_trials;

  double sum_C = 0.0;
  for (int i = 0; i < n * n; ++i){
    sum_C += C[i];
  }
  cout << "Naive sum_C = " << sum_C << endl;

  // reset C
  for (int i = 0; i < n * n; ++i){
    C[i] = 0.0;
  } 

  // Measure performance  
  start = high_resolution_clock::now();
  for (int i = 0; i < num_trials; ++i){  
    matmul_blocked(n, C, A, B, block_size);
  }
  end = high_resolution_clock::now();
  duration<double> elapsed_blocked = (end - start) / num_trials;

  sum_C = 0.0;
  for (int i = 0; i < n * n; ++i){
    sum_C += C[i];
  }  
  cout << "Blocked sum_C = " << sum_C << endl;
  
  cout << "Naive elapsed time (ms) = " << elapsed_naive.count() * 1000 << endl;
  cout << "Blocked elapsed time (ms) = " << elapsed_blocked.count() * 1000 << endl;  

  delete[] A;
  delete[] B;
  delete[] C;  
  
  return 0;
}

