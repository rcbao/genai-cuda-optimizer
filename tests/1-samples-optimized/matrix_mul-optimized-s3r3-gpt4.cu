#include <stdio.h>

// Kernel for matrix multiplication; assumes square matrices
__global__ void matrixMul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index

    float sum = 0.0f;
    if(row < N && col < N) {
        // Utilize shared memory to reduce global memory access
        __shared__ float As[16][16];
        __shared__ float Bs[16][16];

        // Loop over the sub-matrices of A and B required to compute the block sub-matrix
        for (int k = 0; k < (N + 15) / 16; ++k) {
            // Load As and Bs sub-matrices from global memory to shared memory
            // Each thread loads one element of each sub-matrix
            if (k * 16 + threadIdx.x < N && row < N)
                As[threadIdx.y][threadIdx.x] = A[row * N + k * 16 + threadIdx.x];
            else
                As[threadIdx.y][threadIdx.x] = 0.0;

            if (k * 16 + threadIdx.y < N && col < N)
                Bs[threadIdx.y][threadIdx.x] = B[(k * 16 + threadIdx.y) * N + col];
            else
                Bs[threadIdx.y][threadIdx.x] = 0.0;

            // Synchronize to make sure the sub-matrices are loaded
            // before starting the computation
            __syncthreads();

            // Multiply the two sub-matrices together;
            // each thread computes one element of the block sub-matrix
            for (int n = 0; n < 16; ++n)
                sum += As[threadIdx.y][n] * Bs[n][threadIdx.x];

            // Synchronize to make sure that the preceding
            // computation is done before loading two new
            // sub-matrices of A and B in the next iteration
            __syncthreads();
        }

        // Write the block sub-matrix to global memory;
        // each thread writes one element
        if(row < N && col < N)
            C[row * N + col] = sum;
    }
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

    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

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
