#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 128

__global__ void partial_reduction(const int N, float *x_reduced, const float *x)
{

  __shared__ float s_x[BLOCK_SIZE];

  const int i = blockDim.x * 2 * blockIdx.x + threadIdx.x;
  const int tid = threadIdx.x;

  // Coalesced reads in
  s_x[tid] = 0.f;
  if (i < N)
  {
    s_x[tid] = x[i] + x[i + BLOCK_SIZE];
  }

  // Number of "live" threads per block
  int alive = blockDim.x;

  while (alive > 1)
  {
    __syncthreads();
    // Update the number of live threads
    alive /= 2;
    if (tid < alive)
    {
      s_x[tid] += s_x[tid + alive];
    }
  }

  // Write out once we're done reducing each block
  if (tid == 0)
  {
    x_reduced[blockIdx.x] = s_x[0];
  }
}

int main(int argc, char *argv[])
{

  int N = 4194304;
  if (argc > 1)
  {
    N = atoi(argv[1]);
  }

  // Next largest multiple of BLOCK_SIZE
  int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  printf("N = %d, Block Size = %d, Number of Blocks = %d\n", N, BLOCK_SIZE, numBlocks);

  float *x = new float[N];
  float *x_reduced = new float[numBlocks];

  for (int i = 0; i < N; ++i)
  {
    x[i] = 1.f;
  }

  // Allocate memory and copy to the GPU
  float *d_x;
  float *d_x_reduced;
  int size_x = N * sizeof(float);
  int size_x_reduced = numBlocks * sizeof(float);
  cudaMalloc((void **)&d_x, size_x);
  cudaMalloc((void **)&d_x_reduced, size_x_reduced);

  // Copy memory over to the GPU
  cudaMemcpy(d_x, x, size_x, cudaMemcpyHostToDevice);
  cudaMemcpy(d_x_reduced, x_reduced, size_x_reduced, cudaMemcpyHostToDevice);

  partial_reduction<<<numBlocks / 2, BLOCK_SIZE>>>(N, d_x_reduced, d_x);

  // Copy memory back to the CPU
  cudaMemcpy(x_reduced, d_x_reduced, size_x_reduced, cudaMemcpyDeviceToHost);

  // Sum the elements of x_reduced
  float sum_x = 0.f;
  for (int i = 0; i < numBlocks; ++i)
  {
    sum_x += x_reduced[i];
  }

  // Known value of sum of elements
  float target = (float)N;
  printf("error = %f\n", fabs(sum_x - target));

#if 1
  int num_trials = 10;
  cudaEvent_t start, stop;
  float time;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  for (int i = 0; i < num_trials; ++i)
  {
    partial_reduction<<<numBlocks / 2, BLOCK_SIZE>>>(N, d_x_reduced, d_x);
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
  cudaFree(d_x_reduced);

  // Free host memory
  delete[] x;
  delete[] x_reduced;

  return 0;
}