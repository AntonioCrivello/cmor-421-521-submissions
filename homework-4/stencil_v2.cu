#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

// Define size of halo
#define HALO 1

__global__ void stencil(const int N, float *y, const float *x)
{
    // Define shared memory with halo points include
    extern __shared__ float s_x[];

    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int tid = threadIdx.x;
    const int s_tid = tid + HALO;

    if (i < N)
    {
        // Load shared memory for internal cells
        s_x[s_tid] = x[i];

        // Load halo cells into shared memory
        if (tid == 0)
        {
            // Handle the left halo by checking if at the global left boundary
            s_x[s_tid - HALO] = (i == 0) ? x[i] : x[i - HALO];
        }
        if (tid == blockDim.x - 1)
        {
            // Handle the right halo by checking if at the global right boundary
            s_x[s_tid + HALO] = (i == N - 1) ? x[i] : x[i + HALO];
        }

        __syncthreads();

        float xm1 = (i == 0) ? s_x[s_tid] : s_x[s_tid - 1];
        float xp1 = (i == N - 1) ? s_x[s_tid] : s_x[s_tid + 1];
        // Apply stencil operation
        y[i] = -xm1 + 2 * s_x[s_tid] - xp1;
    }
}

int main(int argc, char *argv[])
{
    int N = 4194304;
    if (argc < 2)
    {
        printf("Missing Inputs");
        exit(EXIT_FAILURE);
    }

    // User defined block size
    int blockSize = atoi(argv[1]);

    // Next largest multiple of blockSize
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Define x vector and set all elements to 1
    float *x = new float[N];
    int size_x = (N) * sizeof(float);
    for (int i = 0; i < N; ++i)
    {
        x[i] = 1.f;
    }

    // Define y vector and initialize to all zeros
    float *y = new float[N];
    int size_y = N * sizeof(float);
    for (int i = 0; i < N; ++i)
    {
        y[i] = 0.f;
    }

    // Shared memory size
    int sharedMemSize = (blockSize + 2 * HALO) * sizeof(float);

    // Allocate memory and copy to the GPU
    float *d_x;
    float *d_y;
    cudaMalloc((void **)&d_x, size_x);
    cudaMalloc((void **)&d_y, size_y);

    // Copy memory over to the GPU
    cudaMemcpy(d_x, x, size_x, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size_y, cudaMemcpyHostToDevice);

    stencil<<<numBlocks, blockSize, sharedMemSize>>>(N, d_y, d_x);

    // Copy memory back to the CPU
    cudaMemcpy(y, d_y, size_y, cudaMemcpyDeviceToHost);

    // Known solution of stencil
    float *y_solution = new float[N];
    for (int i = 0; i < N; ++i)
    {
        y_solution[i] = 0.f;
    }

    // Initialize error to zero
    float error = 0.f;
    for (int i = 0; i < N; ++i)
    {
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

    for (int i = 0; i < num_trials; ++i)
    {
        stencil<<<numBlocks, blockSize, sharedMemSize>>>(N, d_y, d_x);
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