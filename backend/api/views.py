import os
import json
from rest_framework.response import Response
from rest_framework import status
from .components.openai_connector import OpenaiConnector
from dotenv import load_dotenv

from django.views import generic
from django.http import HttpResponseRedirect, JsonResponse
from django.urls import reverse
from django.shortcuts import render
from api.models import Settings, CodeComparison
from . import views

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


def return_internal_server_error(error=None):
    if error:
        print("return_internal_server_error: ")
        print("error: ", str(error))
    error_response = {"error": "Internal Server Error"}
    error_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return Response(error_response, status=error_code)


class SettingsView(generic.ListView):
    model = Settings
    template_name = "settings.html"


class CodeComparisonView(generic.ListView):
    model = CodeComparison
    template_name = "code_comparison.html"


def optimize_code(request):
    try:
        """ BACKEND """
        # get POST request data
        # print("got here")
        # data = json.loads(request.body)
        # print("request.POST")
        # print(request.POST)
        # version = data["CUDA_version"]
        # performance = data["speed_rating"]
        # readability = data["readability_rating"]
        # code = data["original_code"]

        """ FRONTEND """
        version = request.POST["CUDA_version"]
        performance = request.POST["speed_rating"]
        readability = request.POST["readability_rating"]
        code = request.POST["original_code"]

        """ USING OPENAI API """
        connector = OpenaiConnector(openai_api_key)
        response = connector.create_newchat(code, version, performance, readability)
        print("view response::", response)
        optimize_code = response["content"]

        if optimize_code.startswith("```cuda\n") and optimize_code.endswith("\n```"):
            optimize_code = optimize_code[len("```cuda\n") : -len("\n```")]

        if "error" in response:
            return JsonResponse(response, status=status.HTTP_400_BAD_REQUEST)
        else:
            return render(
                request,
                "code_comparison.html",
                {"original_code": code, "optimized_code": optimize_code},
            )
        
        """ BEGINNING OF USING CONSTANTS """
        # optimize_code = """__global__ void gpuMM(float *A, float *B, float *C, int N)
        # {
        #     // Optimized Matrix multiplication for NxN matrices C=A*B using shared memory
        #     // Each thread computes a single element of C
        #     int row = blockIdx.y * blockDim.y + threadIdx.y;
        #     int col = blockIdx.x * blockDim.x + threadIdx.x;
        #     __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        #     __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        #     float sum = 0.0f;
        #     int numBlocks = N / BLOCK_SIZE;
        #     for (int block = 0; block < numBlocks; ++block) {
        #         // Load A and B matrices into shared memory
        #         As[threadIdx.y][threadIdx.x] = A[row * N + (block * BLOCK_SIZE + threadIdx.x)];
        #         Bs[threadIdx.y][threadIdx.x] = B[(block * BLOCK_SIZE + threadIdx.y) * N + col];
        #         __syncthreads(); // Ensure all threads have loaded their parts before computation
        #         // Compute partial product for the block
        #         for (int k = 0; k < BLOCK_SIZE; ++k) {
        #             sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        #         }
        #         __syncthreads(); // Ensure all threads have finished computation before loading next block
        #     }
        #     // Write the computed element to the matrix C
        #     if (row < N && col < N) {
        #         C[row * N + col] = sum;
        #     }
        # }"""

        # code = """
        # // source: https://github.com/emandere/CudaProject/blob/master/cudamatrix.cu
        # #include<ctime>
        # #include<iostream>
        # using namespace std;
        # #define BLOCK_SIZE 32
        # __global__ void gpuMM(float *A, float *B, float *C, int N)
        # {
        #     // Matrix multiplication for NxN matrices C=A*B
        #     // Each thread computes a single element of C
        #     int row = blockIdx.y*blockDim.y + threadIdx.y;
        #     int col = blockIdx.x*blockDim.x + threadIdx.x;
        #     float sum = 0.f;
        #     for (int n = 0; n < N; ++n)
        #         sum += A[row*N+n]*B[n*N+col];
        #     C[row*N+col] = sum;
        # }
        # int testmatrix(int K)
        # {
        #     // Perform matrix multiplication C = A*B
        #     // where A, B and C are NxN matrices
        #     // Restricted to matrices where N = K*BLOCK_SIZE;
        #     int N;
        #     N = K*BLOCK_SIZE;
            
        #     //cout << "Executing Matrix Multiplcation" << endl;
        #     //cout << "Matrix size: " << N << "x" << N << endl;
        #     // Allocate memory on the host
        #     float *hA,*hB,*hC;
        #     hA = new float[N*N];
        #     hB = new float[N*N];
        #     hC = new float[N*N];
        #     // Initialize matrices on the host
        #     for (int j=0; j<N; j++){
        #         for (int i=0; i<N; i++){
        #             hA[j*N+i] = 1.0f;//2.f*(j+i);
        #             hB[j*N+i] = 1.0f;//1.f*(j-i);
        #         }
        #     }
        #     // Allocate memory on the device
        #     long size = N*N*sizeof(float);	// Size of the memory in bytes
        #     float *dA,*dB,*dC;
        #     cudaMalloc(&dA,size);
        #     cudaMalloc(&dB,size);
        #     cudaMalloc(&dC,size);
        #     dim3 threadBlock(BLOCK_SIZE,BLOCK_SIZE);
        #     dim3 grid(K,K);
            
        #     // Copy matrices from the host to device
        #     cudaMemcpy(dA,hA,size,cudaMemcpyHostToDevice);
        #     cudaMemcpy(dB,hB,size,cudaMemcpyHostToDevice);
            
        #     //Execute the matrix multiplication kernel
            
        #     gpuMM<<<grid,threadBlock>>>(dA,dB,dC,N);
                
            
        #     // Allocate memory to store the GPU answer on the host
        #     float *C;
        #     C = new float[N*N];
            
        #     // Now copy the GPU result back to CPU
        #     cudaMemcpy(C,dC,size,cudaMemcpyDeviceToHost);
        #     cudaFree( dA );
        #     cudaFree( dB );
        #     cout<<"N "<<N<<" C[0][0] "<<C[0]<<endl;
            
            
        # }
        # int main()
        # {
        #     const int matrix_size = 5000;
        #     clock_t start;
        #     double duration;
        #     for (int i = 140; i < 150 ; i++)
        #     {
        #         start = std::clock();
        #         testmatrix(i);
        #         duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
        #         cout <<i<< " " << duration <<"s"<< '\n';
        #     }
            
        #     return 0;
        # }
        # """
        
        """ END OF USING CONSTANTS """

        return render(
                request,
                "code_comparison.html",
                {"original_code": code, "optimized_code": optimize_code},
            )

    except KeyError as e:
        return JsonResponse(
            {"error": f"Missing parameter: {e}"}, status=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        return JsonResponse(
            {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


def back(request):
    return HttpResponseRedirect(reverse("settings"))
