#include <cuda_runtime.h>
#include <iostream>

using namespace std;

#define BLOCK_SIZE 32
#define WARP_SIZE 32

// Warp-level matrix multiplication kernel for maximum performance
__global__ void warpOptimizedGpuMM(float *A, float *B, float *C, int N) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = N * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + N - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * N;

    float Csub = 0;

    // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
        As[ty][tx] = A[a + N * ty + tx];
        Bs[ty][tx] = B[b + N * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // Each thread computes one element of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory; each thread writes one element
    int c = N * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + N * ty + tx] = Csub;
}

void testMatrixMultiplication(int K) {
    int N = K * BLOCK_SIZE;
    float *hA, *hB, *hC;
    float *dA, *dB, *dC;
    size_t size = N * N * sizeof(float);

    // Allocate host memory
    hA = new float[N*N];
    hB = new float[N*N];
    hC = new float[N*N];

    // Initialize matrices
    for (int i = 0; i < N*N; i++) {
        hA[i] = 1.0f; // Simplified initialization for demonstration
        hB[i] = 1.0f; // Simplified initialization for demonstration
    }

    // Allocate device memory
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    // Copy host memory to device
    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    // Setup execution configuration
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

    // Execute the kernel
    warpOptimizedGpuMM<<<dimGrid, dimBlock>>>(dA, dB, dC, N);

    // Copy result back to host
    cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    // Free host memory
    delete[] hA;
    delete[] hB;
    delete[] hC;
}

int main() {
    const int matrix_size = 5000; // This specific value isn't used due to the loop condition

    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 140; i < 150; i++) {
        cudaEventRecord(start);

        testMatrixMultiplication(i); // Adapted to match the optimized function's signature

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cout << "Matrix size: " << i * BLOCK_SIZE << ", Time: " << milliseconds / 1000.0 << "s" << endl;
    }

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
