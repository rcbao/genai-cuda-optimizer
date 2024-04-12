#include <stdio.h>
#include <cuda_runtime.h>

// Optimized kernel for adding two vectors using loop unrolling and increased parallelism
__global__ void vectorAdd(int *A, int *B, int *C, int numElements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Using grid-stride loop to handle any size of data
    for (int index = i; index < numElements; index += stride) {
        C[index] = A[index] + B[index];
    }
}

// Main function to set up and execute vector addition
int main() {
    int numElements = 1024; // Number of elements in each vector
    int size = numElements * sizeof(int);
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;

    // Allocate memory on the host
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);

    // Initialize input vectors
    for(int i = 0; i < numElements; i++) {
        h_A[i] = rand() % 100;
        h_B[i] = rand() % 100;
    }

    // Allocate vectors in device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch the Vector Addition CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Record the start event
    cudaEventRecord(start, 0);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // Record the stop event
    cudaEventRecord(stop, 0);

    // Copy the result from device memory to host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Calculate and print the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time: %f ms\n", milliseconds);

    // Free device and host memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}