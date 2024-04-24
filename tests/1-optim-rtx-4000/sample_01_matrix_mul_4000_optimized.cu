#include <stdio.h>
#include <cuda_runtime.h>

// Optimized kernel for matrix multiplication using shared memory and loop unrolling
__global__ void matrixMul(int *A, int *B, int *C, int width) {
    // Define block size and tile size
    const int BLOCK_SIZE = 16;
    __shared__ int s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int s_B[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    int sum = 0;

    // Loop over tiles of the input matrices
    for (int m = 0; m < (width / BLOCK_SIZE); ++m) {
        // Load tiles into shared memory
        s_A[ty][tx] = A[row * width + (m * BLOCK_SIZE + tx)];
        s_B[ty][tx] = B[(m * BLOCK_SIZE + ty) * width + col];
        __syncthreads();

        // Compute product for the tile
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }
        __syncthreads();
    }

    // Write the block's results to the output matrix
    if (row < width && col < width) {
        C[row * width + col] = sum;
    }
}

// Main function to set up matrices and call the kernel
int main() {
    int width = 32; // Small size for simplicity
    int size = width * width;
    int bytes = size * sizeof(int);
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;

    // Allocate memory on the host
    h_A = (int*)malloc(bytes);
    h_B = (int*)malloc(bytes);
    h_C = (int*)malloc(bytes);

    // Initialize matrices on the host
    for(int i = 0; i < size; i++) {
        h_A[i] = rand() % 1024;
        h_B[i] = rand() % 1024;
        h_C[i] = 0;
    }

    // Allocate memory on the device
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Execute the kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + 15) / 16, (width + 15) / 16);

    // Record the start event
    cudaEventRecord(start, 0);

    matrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, width);

    // Record the stop event
    cudaEventRecord(stop, 0);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time: %f ms\n", milliseconds);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}