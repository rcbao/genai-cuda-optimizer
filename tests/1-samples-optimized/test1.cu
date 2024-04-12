// source: https://github.com/emandere/CudaProject/blob/master/cudamatrix.cu

#include<ctime>
#include<iostream>
using namespace std;

#define BLOCK_SIZE 32
#define TILE_WIDTH 32

__global__ void gpuMM(float *A, float *B, float *C, int N)
{
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the Pd element to work on
    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;

    float Pvalue = 0;

    // Loop over the Md and Nd tiles required to compute the Pd element
    for (int m = 0; m < (N-1)/TILE_WIDTH+1; ++m) {
        // Collaborative loading of Md and Nd tiles into shared memory
        __shared__ float As[TILE_WIDTH][TILE_WIDTH];
        __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
        if (Row < N && (m*TILE_WIDTH + tx) < N)
            As[ty][tx] = A[Row*N + m*TILE_WIDTH + tx];
        else
            As[ty][tx] = 0.0;
        if (Col < N && (m*TILE_WIDTH + ty) < N)
            Bs[ty][tx] = B[(m*TILE_WIDTH + ty)*N + Col];
        else
            Bs[ty][tx] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    if (Row < N && Col < N) C[Row*N + Col] = Pvalue;
}

int testmatrix(int K)
{
    // Perform matrix multiplication C = A*B
    // where A, B and C are NxN matrices
    // Restricted to matrices where N = K*BLOCK_SIZE;
    int N;
    N = K*BLOCK_SIZE;
    
    //cout << "Executing Matrix Multiplcation" << endl;
    //cout << "Matrix size: " << N << "x" << N << endl;

    // Allocate memory on the host
    float *hA,*hB,*hC;
    hA = new float[N*N];
    hB = new float[N*N];
    hC = new float[N*N];

    // Initialize matrices on the host
    for (int j=0; j<N; j++){
        for (int i=0; i<N; i++){
            hA[j*N+i] = 1.0f;//2.f*(j+i);
            hB[j*N+i] = 1.0f;//1.f*(j-i);
        }
    }

    // Allocate memory on the device
    long size = N*N*sizeof(float);    // Size of the memory in bytes
    float *dA,*dB,*dC;
    cudaMalloc(&dA,size);
    cudaMalloc(&dB,size);
    cudaMalloc(&dC,size);

    dim3 threadBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 grid(K,K);
    
    // Copy matrices from the host to device
    cudaMemcpy(dA,hA,size,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB,size,cudaMemcpyHostToDevice);
    
    //Execute the matrix multiplication kernel
    
    gpuMM<<<grid,threadBlock>>>(dA,dB,dC,N);
        
    
    // Allocate memory to store the GPU answer on the host
    float *C;
    C = new float[N*N];
    
    // Now copy the GPU result back to CPU
    cudaMemcpy(C,dC,size,cudaMemcpyDeviceToHost);
    cudaFree( dA );
    cudaFree( dB );
    cout<<"N "<<N<<" C[0][0] "<<C[0]<<endl;
    
    
}



int main()
{
    const int matrix_size = 5000;

    clock_t start;
    double duration;

    for (int i = 140; i < 150 ; i++)
    {
        start = std::clock();
        testmatrix(i);
        duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
        cout <<i<< " " << duration <<"s"<< '\n';
    }
    
    return 0;
}
