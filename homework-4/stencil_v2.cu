#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void stencil(const int N, float * y, const float * x) {

    __shared__ float s_x[blockSize + 2];

    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int tid = threadIdx.x;

    // Coalesced reads in
    s_x[tid] = 0.f;
    if (i < N) {
        s_x[tid] = x[i];
    }

    y[i] = -1 * s_x[i - 1] + 2 * s_x[i] - s_x[i - 1]

}

int main(int argc, char * argv[]) {
    int N = 4194304;
    if (argc < 2) {
        printf("Missing Inputs")
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

    stencil <<< numBlocks, blockSize >>> (N, d_y, d_x)
}