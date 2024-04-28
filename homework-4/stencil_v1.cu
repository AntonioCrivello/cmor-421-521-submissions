#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void stencil(const int N, float * y, const float * x) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
      float xm1, xn, xp1;
      xn = x[i];
      // Handle left boundary condition
      xm1 = (i == 0) ? xn : x[i - 1];
      // Handle right boundary condition
      xp1 = (i == N - 1) ? xn : x[i + 1];
      
      __syncthreads();

      // Apply stencil operation 
      y[i] = -1 * xp1 + 2 * xn - xm1; 
    }
}

int main(int argc, char * argv[]) {
    int N = 4194304;
    if (argc < 2) {
        printf("Missing Inputs");
        exit(EXIT_FAILURE);
    }

    // User defined block size
    int blockSize = atoi(argv[1]);

    // Next largest multiple of blockSize
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Define x vector and set all elements to 1
    float *x = new float [N];
    int size_x = (N) * sizeof(float);
    for (int i = 0; i < N; ++i) {
        x[i] = 1.f;
    }

    // Define y vector and initialize to all zeros
    float *y = new float [N];
    int size_y = N * sizeof(float);
    for (int i = 0; i < N; ++i) {
        y[i] = 0.f;
    }

    // Allocate memory and copy to the GPU
    float *d_x;
    float *d_y;
    cudaMalloc((void **) &d_x, size_x);
    cudaMalloc((void **) &d_y, size_y);

    // Copy memory over to the GPU
    cudaMemcpy(d_x, x, size_x, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size_y, cudaMemcpyHostToDevice);

    // Execute stencil kernel
    stencil <<< numBlocks, blockSize >>> (N, d_y, d_x);

    // Copy memory back to the CPU
    cudaMemcpy(y, d_y, size_y, cudaMemcpyDeviceToHost);

    // Known solution of stencil
    float *y_solution = new float[N];
    for (int i = 0; i < N; ++i) {
        y_solution[i] = 0.f;
    }

    // Initialize error to zero
    float error = 0.f;
    for (int i = 0; i < N; ++i) {
      error += fabs(y[i] - y_solution[i]);
    }
    printf("error = %f\n", error);

#if 1
  int num_trials = 10;
  cudaEvent_t start, stop;
  float time;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  for (int i = 0; i < num_trials; ++i){
    stencil <<< numBlocks, blockSize >>> (N, d_y, d_x);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  float average_time = time / num_trials;

  printf("Time to run kernel on average: %6.6f ms.\n", average_time);
  
#endif

  // Free device memory
  cudaFree(d_x);
  cudaFree(d_y);

  // Free host memory
  delete[] x;
  delete[] y;
  delete[] y_solution;

  return 0;
}