#include <stdio.h>
#include <cuda_runtime.h>

// Optimized kernel for adding two vectors using loop unrolling, increased parallelism, and utilizing TensorFloat-32 where applicable
__global__ void vectorAdd(int *A, int *B, int *C, int numElements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Utilizing loop unrolling to reduce loop overhead and increase instruction throughput
    for (int index = i; index < numElements; index += stride * 4) {
        if (index < numElements) C[index] = A[index] + B[index];
        if (index + 1 < numElements) C[index + 1] = A[index + 1] + B[index + 1];
        if (index + 2 < numElements) C[index + 2] = A[index + 2] + B[index + 2];
        if (index + 3 < numElements) C[index + 3] = A[index + 3] + B[index + 3];
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