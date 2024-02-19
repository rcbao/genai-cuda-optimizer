#include <cuda_runtime.h>
#include <iostream>

using namespace std;

#define BLOCK_SIZE 32 // Assuming 32x32 is still a good blockDim size for modern GPUs.

__global__ void gpuMMShared(float *A, float *B, float *C, int N) {
    int bx = blockIdx.x, by = blockIdx.y,
        tx = threadIdx.x, ty = threadIdx.y,
        Row = by * blockDim.y + ty,
        Col = bx * blockDim.x + tx;
    
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;
    for (int ph = 0; ph < ceil(N/(float)BLOCK_SIZE); ++ph) {
        if(Row < N && ph*BLOCK_SIZE + tx < N)
            sA[ty][tx] = A[Row*N + ph*BLOCK_SIZE + tx];
        else
            sA[ty][tx] = 0.0;
        
        if(Col < N && ph*BLOCK_SIZE + ty < N)
            sB[ty][tx] = B[(ph*BLOCK_SIZE + ty)*N + Col];
        else
            sB[ty][tx] = 0.0;
        
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }

    if(Row < N && Col < N)
        C[Row*N + Col] = sum;
}

int testmatrix(int K) {
    int N = K * BLOCK_SIZE;
    long size = N * N * sizeof(float);
    float *hA, *hB, *hC, *dA, *dB, *dC;

    // Allocate & initialize host memory
    hA = new float[N*N];
    hB = new float[N*N];
    hC = new float[N*N];
    for (int i = 0; i < N*N; ++i) {
        hA[i] = 1.0f;
        hB[i] = 1.0f;
    }

    // Allocate device memory
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    // Copy host memory to device
    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    // Setup execution configuration
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Execute the kernel
    gpuMMShared<<<grid, threads>>>(dA, dB, dC, N);

    // Copy result back to host
    cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);
    cout << "N " << N << " C[0][0] " << hC[0] << endl;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    delete[] hA;
    delete[] hB;
    delete[] hC;

    return 0; // Return success status
}

int main() {
    const int matrix_size = 5000;

    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 140; i < 150; i++) {
        cudaEventRecord(start);
        
        testmatrix(i);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cout << i << " " << milliseconds / 1000.0 << "s" << '\n';
    }

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}