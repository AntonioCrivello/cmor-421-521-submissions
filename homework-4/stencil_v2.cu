#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

// Define size of halo
#define HALO 1

__global__ void stencil(const int N, float *y, const float *x) {

    // Define shared memory with halo points include
    __shared__ float s_x[blockSize + 2];

    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int tid = threadIdx.x ;

    if (i < N) {
      s_x[tid + HALO] = x[i];

      s_x[tid] = (tid < HALO) ? x[i - HALO] : x[i];
      s_x[tid + blockDim. + HALO] ? x[i + blockDim.x + HALO] : x[i];

      __syncthreads();

      

    }





    // s_x = 0.f;
    // s_x[tid] = (tid < HALO) ? x[i - HALO] : x[i];
    // s_x[tid + blockSize] = (tid < HALO) ? x[i + blockSize + HALO]: x[i];

    // s_xn1 = (i == 0 && tid == 0) ? s_x[i - HALO] : s_x[tid + HALO - 1];
    // s_xp1 = (i == N - 1 && tid == blockSize + 2) ? s_x[i + HALO] : s_x[tid + blockSize]

    // s_x = 0.f;
    // // Populate shared memory
    // if (tid < HALO) {
    //     // Left halo point for shared memory
    //     s_x[tid] = x[i - HALO];
    //     // Right halo point for shared memory
    //     s_x[tid + blockSize] = x[i + HALO];
    // }
    // // Internal shared memory points
    // s_x[tid + HALO] = x[i];

    // Synchronize threads
    __syncthreads();

    // Compute y[i]
    y[i] = -1 * s_x[tid + HALO - 1] + 2 * s_x[tid + HALO] - s_x[tid + HALO + 1];
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

    // x vector
    float * x = new float [N];
    int size_x = N * sizeof(float);

    // y vector
    float * y = new float [N];
    int size_y = N * sizeof(float);

    // Defining x_i = 1
    for (int i = 0; i < N; ++i) {
        x[i] = 1.f;
    }

    // Allocate memory and copy to the GPU
    float * d_x;
    float * d_y;
    cudaMalloc((void **) &d_x, size_x);
    cudaMalloc((void **) &d_y, size_y);

    // Copy memory over to the GPU
    cudaMemcpy(d_x, x, size_x, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size_y, cudaMemcpyHostToDevice);

    stencil <<< numBlocks, blockSize >>> (N, d_y, d_x);

    // Copy memory back to the CPU
    cudaMemcpy(y, d_y, size_y, cudaMemcpyHostToDevice);

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