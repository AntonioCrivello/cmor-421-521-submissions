#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void stencil(const int N, float * y, const float * x) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        printf("i = %d\n", i);
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

  // Known solution of stencil
  float * y_solution = new float[N];
  for (int i = 0; i < N; ++i) {
        y_solution[i] = 1.f;
  }

  // Initialize error to zero
  int error = 0;
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
    stencil <<< numBlocks, blockSize >>> (N, d_y, x);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  float average_time = time / num_trials;

  printf("Time to run kernel on average: %6.6f ms.\n", average_time);
  
#endif

  return 0;
}