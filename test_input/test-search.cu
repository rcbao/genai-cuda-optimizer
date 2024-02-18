// https://github.com/emandere/CudaProject/blob/master/testSearch.cu

#include<ctime>
#include <cmath>
#include<iostream>
#include <cstdlib>
using namespace std;

#define BLOCK_SIZE 1024

__global__ void gpuSum(int *prices,int *sumpricesout,int days,int seconds,int N)
{
    int currentday = blockIdx.x*blockDim.x + threadIdx.x;
    if(currentday<days)
    {
       int start = currentday * seconds;
       int end = start+seconds;
       
       int totprice=0; 
       for(int j=start;j<end;++j)
            totprice+=prices[j];
        
       sumpricesout[currentday] = totprice;
    }
}

int main()
{
   int days =1200000;
   int seconds = 1000;
   
   clock_t start;
   double duration; 
   start = std::clock();
    
   int N = days*seconds;
   
   int * prices = new int[days*seconds];
   int  * sumpricesout = new int[days];
   int  * sumpricesoutCPU = new int[days]; 
  
   for(int i=0;i<N;i++)
   {
      prices[i]=rand()%100;  
   }
   
  
   for(int i=0;i<N;i=i+seconds)
   {
      for(int j=i;j<i+seconds;j++)
         sumpricesoutCPU[i/seconds]+=prices[j];
      
   }
    duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
    cout<<"CPU: "<< duration <<"s"<< '\n'; 
    
   /*for(int i=0;i<days;i++)
   {
       cout<<sumpricesoutCPU[i]<<endl;
   }*/
    
   long sizePrices = N * sizeof(int);
   long sizeSumPrices = days * sizeof(int); 
   int *dPrices,*dSumPrices;
   
   start = std::clock(); 
   cudaMalloc(&dPrices,sizePrices);
   cudaMalloc(&dSumPrices,sizeSumPrices);
    
   cudaMemcpy(dPrices,prices,sizePrices,cudaMemcpyHostToDevice); 
   cudaMemcpy(dSumPrices,sumpricesout,sizeSumPrices,cudaMemcpyHostToDevice); 
    
   gpuSum<<<(int)ceil(days/(float)BLOCK_SIZE),BLOCK_SIZE>>>(dPrices,dSumPrices,days,seconds,N);
   
   cudaMemcpy(sumpricesout,dSumPrices,sizeSumPrices,cudaMemcpyDeviceToHost); 
   
   duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
   cout<<"GPU: "<< duration <<"s"<< '\n';  
   //cout<<"CUDA!"<<endl;
   /*for(int i=0;i<days;i++)
   {
       cout<<sumpricesout[i]<<endl;
   }*/
    
   int error = 0;
   for(int i=0;i<days;i++)
   {
       error+=sumpricesout[i] - sumpricesoutCPU[i];
   }
    cout<<"Error: "<< error<<endl; 
    //cout<<(int)ceil(days/(float)BLOCK_SIZE)<<endl;
   cudaFree(dPrices);
   cudaFree(dSumPrices); 
   return 0;
}

