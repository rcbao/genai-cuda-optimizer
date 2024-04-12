
#include <vector>
#include <iterator>
#include <fstream>
#include<iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <numeric>
#include<cuda.h>
#include<time.h>
#define N 50
#define M 10
#define SHARED_DATA 10*4
using namespace std;
static const int ArraySize = 500000000;
static const int BlockSize = 1024;
static const int GridSize = 24;
static const int arraySize = 1000000000;


__global__ void parallelSum(const int *input, int arraySize, int *output) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;
    sdata[tid] = 0;
    while (i < arraySize) {
        sdata[tid] += input[i];
        i += gridSize;
    }
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

__global__ void sum(int *input, int *output) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < N) ? input[i] : 0;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

int main() {
	 int size=N*sizeof(int);
	clock_t startLoad50,startMove50,startSum50;
	clock_t endLoad50,endMove50,endSum50;
	startLoad50=clock();
	std::ifstream file_handler("/mirror/data/50Numbers.txt");
	
	std::vector<int> input1;
	int number;
	while (file_handler>>number) {
  		input1.push_back(number);
        }
	endLoad50=clock();
	int *output1;
	output1=(int*)malloc(size);
 
 	int *d_input1, *d_sum1; //Device variable Declaration

	cudaMalloc((void **)&d_input1, size);
	cudaMalloc((void **)&d_sum1, size);
 	startMove50=clock();
	cudaMemcpy(d_input1, input1.data(), size,cudaMemcpyHostToDevice);
	endMove50=clock();
	cudaMemcpy(d_sum1, output1, size, cudaMemcpyHostToDevice);
  
	//Launch Kernel
	startSum50=clock();
	 sum << <(N+M-1)/M,M >> >(d_input1,d_sum1);
	sum << <1,N >> >(d_input1,d_sum1);
	endSum50=clock();


 //Copy Device Memory to Host Memory
 cudaMemcpy(output1, d_sum1, sizeof(int), cudaMemcpyDeviceToHost);
printf("50Numbers:\n");
printf("Time consumption to load the File:%f secs\n", (double) (endLoad50-startLoad50)/ CLOCKS_PER_SEC);
printf("Time consumption to move the file from main memory to device memory:%f secs \n", (double) (endMove50-startMove50)/ CLOCKS_PER_SEC);
printf("Time consumption to sum the file:%f secs\n", (double) (endSum50-startSum50)/ CLOCKS_PER_SEC);
printf("Sum is:%d\n",output1[0]);
printf("\n");

	free(output1);
        //Free Device Memory
 	cudaFree(&d_input1);
        cudaFree(&d_sum1);



clock_t startLoadHalfBillion,startMoveHalfBillion,startSumHalfBillion;
clock_t endLoadHalfBillion,endMoveHalfBillion,endSumHalfBillion;
startLoadHalfBillion=clock();
std::ifstream file_handler1("/mirror/data/halfBillionNumbers.txt");
	std::vector<int> input2;
	int number1;
	while (file_handler1>>number1) {
  		input2.push_back(number1);
        }
endLoadHalfBillion=clock();
    int* d_input2,*d_output2;
    int output2;
    cudaMalloc((void**)&d_input2, ArraySize*sizeof(int));
   cudaMalloc((void**)&d_output2, sizeof(int)*GridSize);
startMoveHalfBillion=clock();
    cudaMemcpy(d_input2, input2.data(), ArraySize*sizeof(int), cudaMemcpyHostToDevice);
   endMoveHalfBillion=clock();
    startSumHalfBillion=clock();
    parallelSum<<<GridSize, BlockSize>>>(d_input2, ArraySize, d_output2);
    
    parallelSum<<<1, BlockSize>>>(d_output2, GridSize, d_output2);
    endSumHalfBillion=clock();
    cudaDeviceSynchronize();
    cudaMemcpy(&output2, d_output2, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_input2);
    cudaFree(d_output2);


printf("halfBillionNumbers:\n");
printf("Time consumption to load the File:%f secs\n", (double) (endLoadHalfBillion-startLoadHalfBillion)/ CLOCKS_PER_SEC);
printf("Time consumption to move the file from main memory to device memory:%f secs\n", (double) (endMoveHalfBillion-startMoveHalfBillion)/ CLOCKS_PER_SEC);
printf("Time consumption to sum the file:%f secs\n", (double) (endSumHalfBillion-startSumHalfBillion)/ CLOCKS_PER_SEC);
printf("Sum is:%d\n",output2);
printf("\n");


clock_t startLoadBillion,startMoveBillion,startSumBillion;
clock_t endLoadBillion,endMoveBillion,endSumBillion;
startLoadBillion=clock();
std::ifstream file_handler2("/mirror/data/1billionNumbers.txt");
	std::vector<int> input3;
	int number2;
	while (file_handler2>>number2) {
  		input3.push_back(number2);
        }
endLoadBillion=clock();
    int* d_input3,*d_output3;
    int output3;
    cudaMalloc((void**)&d_input3, arraySize*sizeof(int));
   cudaMalloc((void**)&d_output3, sizeof(int)*GridSize);
startMoveBillion=clock();
    cudaMemcpy(d_input3, input3.data(), arraySize*sizeof(int), cudaMemcpyHostToDevice);
    endMoveBillion=clock();
    startSumBillion=clock();
    parallelSum<<<GridSize, BlockSize>>>(d_input3, arraySize, d_output3);
    
    parallelSum<<<1, BlockSize>>>(d_output3, GridSize, d_output3);
    endSumBillion=clock();
    cudaDeviceSynchronize();
    cudaMemcpy(&output3, d_output3, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_input3);
    cudaFree(d_output3);


printf("1billionNumbers:\n");
printf("Time consumption to load the File:%f secs\n", (double) (endLoadBillion-startLoadBillion)/ CLOCKS_PER_SEC );
printf("Time consumption to move the file from main memory to device memory:%f secs\n", (double) (endMoveBillion-startMoveBillion)/ CLOCKS_PER_SEC);
printf("Time consumption to sum the file:%f secs\n", (double) (endSumBillion-startSumBillion)/ CLOCKS_PER_SEC);
printf("Sum is:%d\n",output3);
printf("\n");
      

return 0;
	
}

