#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_WIDTH 32

__global__ void matrixMulOptimized(float *A, float *B, float *C, int N) {
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float sum = 0.0f;

    for (int m = 0; m < (N / TILE_WIDTH); ++m) {
        s_A[ty][tx] = A[row * N + m * TILE_WIDTH + tx];
        s_B[ty][tx] = B[(m * TILE_WIDTH + ty) * N + col];
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }
        __syncthreads();
    }
    if(row < N && col < N) C[row * N + col] = sum;
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
