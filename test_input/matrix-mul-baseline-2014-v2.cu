// prompt used: Write CUDA accelerated code to perform matrix multiplication. Ensure compatibility to a CUDA version in 2014. 
// model: gpt-4-turbo

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel for performing simple matrix multiplication
__global__ void simpleMatrixMul(float *C, float *A, float *B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;
    if (row < N && col < N) {
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int N = 256; // Matrix size N x N
    size_t size = N * N * sizeof(float);
    float *A, *B, *C, *d_A, *d_B, *d_C;

    // Allocate host memory
    A = (float *)malloc(size);
    B = (float *)malloc(size);
    C = (float *)malloc(size);

    // Initialize matrices A and B
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f; // Example value
        B[i] = 2.0f; // Example value
    }

    // Allocate device memory
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy matrices from the host to the device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Define block size and grid size
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    // Launch the kernel
    simpleMatrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, N);

    // Copy the result matrix back to the host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C);

    return 0;
}

