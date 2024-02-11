// prompt used: Write CUDA accelerated code to perform matrix multiplication. Ensure compatibility to a CUDA version in 2014. 
// model: gpt-4-turbo

#include <stdio.h>
#include <cuda.h>

#define TILE_WIDTH 16 // Assuming a small tile width for simplicity

__global__ void matrixMulCUDA(float *C, float *A, float *B, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (row < width && col < width) {
        for (int k = 0; k < width; ++k) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

int main() {
    int width = 256; // Example size
    size_t size = width * width * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize matrices A and B with some values
    for (int i = 0; i < width*width; ++i) {
        h_A[i] = 1.0f; // Example initialization
        h_B[i] = 2.0f; // Example initialization
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (width + dimBlock.y - 1) / dimBlock.y);

    matrixMulCUDA<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, width);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Use h_C for result

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

