#include <cuda_runtime.h>
#include <iostream>

// Size of the square matrices
#define N 512
#define BLOCK_SIZE 16 // Size of the block, can be tuned for performance

// CUDA Kernel for Matrix Multiplication
__global__ void matrixMulCUDA(float *C, const float *A, const float *B, int n) {
    // Calculate the row index of the C matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column index of the C matrix
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes one element of the C matrix by accumulating results into sum
    if(row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    // Host memory allocation
    float *h_A, *h_B, *h_C;
    size_t bytes = N * N * sizeof(float);
    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_C = (float*)malloc(bytes);

    // Initialize matrices with test data
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 0.01f;
        h_B[i] = 0.02f;
    }

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // Execute the matrix multiplication kernel
    matrixMulCUDA<<<grid, block>>>(d_C, d_A, d_B, N);

    // Copy the result matrix from device to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

