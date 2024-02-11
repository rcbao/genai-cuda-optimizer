#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_WIDTH 32 // Adjust based on the compute capability

// Optimized kernel using shared memory and tiling
__global__ void matrixMulOptimized(float *C, float *A, float *B, int N) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float sum = 0.0f;

    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    for (int ph = 0; ph < ceil(N/(float)TILE_WIDTH); ++ph) {
        if (row < N && ph*TILE_WIDTH+tx < N)
            As[ty][tx] = A[row*N + ph*TILE_WIDTH + tx];
        else
            As[ty][tx] = 0.0;
        if (col < N && ph*TILE_WIDTH+ty < N)
            Bs[ty][tx] = B[(ph*TILE_WIDTH + ty)*N + col];
        else
            Bs[ty][tx] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    if (row < N && col < N)
        C[row*N + col] = sum;
}

int main() {
    // Same setup and teardown code as before, with the kernel call updated to use matrixMulOptimized
}

