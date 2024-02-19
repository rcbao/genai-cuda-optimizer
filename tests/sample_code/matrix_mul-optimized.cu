#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_WIDTH 32
#define WARP_SIZE 32
#define K 8 // Loop unrolling factor

__global__ void matrixMulOptimized(float *A, float *B, float *C, int N) {
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Identify the row and column of the C element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float CValue = 0;

    // Loop over the A and B tiles required to compute the C element
    for (int ph = 0; ph < ceilf(N / (float)TILE_WIDTH); ++ph) {
        // Collaborative loading of A and B tiles into shared memory
        if (Row < N && ph*TILE_WIDTH+tx < N)
            sA[ty][tx] = A[Row*N + ph*TILE_WIDTH+tx];
        else
            sA[ty][tx] = 0.0;

        if (Col < N && ph*TILE_WIDTH+ty < N)
            sB[ty][tx] = B[(ph*TILE_WIDTH+ty)*N + Col];
        else
            sB[ty][tx] = 0.0;

        __syncthreads();

        // Computation loop unrolled
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; k += K) {
            CValue += sA[ty][k] * sB[k][tx];
            // Include further unrolled computations
            CValue += sA[ty][k] * sB[k][tx];
            CValue += sA[ty][k+1] * sB[k+1][tx];
            CValue += sA[ty][k+2] * sB[k+2][tx];
            CValue += sA[ty][k+3] * sB[k+3][tx];
            CValue += sA[ty][k+4] * sB[k+4][tx];
            CValue += sA[ty][k+5] * sB[k+5][tx];
            CValue += sA[ty][k+6] * sB[k+6][tx];
            CValue += sA[ty][k+7] * sB[k+7][tx];
        }

        __syncthreads();
    }

    if (Row < N && Col < N)
        C[Row*N + Col] = CValue;
}

int main() {
    int N = 65536; // Matrix size (N x N)
    size_t size = N * N * sizeof(float);
    float *A, *B, *C; // Host matrices
    float *d_A, *d_B, *d_C; // Device matrices

    // Allocate host memory
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    // Initialize matrices A and B on the host
    for(int i = 0; i < N*N; i++) {
        A[i] = 1.0; // Simplified initialization
        B[i] = 2.0;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy host memory to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Launch the kernel with a single thread per block for maximum inefficiency
    dim3 threadsPerBlock(1, 1);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMulOptimized<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);

    return 0;
}
