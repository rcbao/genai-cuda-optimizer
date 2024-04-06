import re
from .gpt_runner import GPTRunner
from .prompt_builder import PromptBuilder
from ..constants import LIGHTWEIGHT_OPENAI_MODEL


class CudaMarkdownarser:

    def parse(llm_response: dict[str, str]) -> str:

        markdown_text = llm_response["content"]

        codeblock_exists = "```cuda" in markdown_text

        if codeblock_exists:
            pattern = r"```cuda\n(.*?)```"
            match = re.search(pattern, markdown_text, re.DOTALL)

            return str(match.group(1)) if match else None
        return markdown_text


class SignatureDiff:
    def __init__(self, original_code: str, optimized_code: str):
        self.original_code = original_code
        self.optimized_code = optimized_code

    def extract_kernel_signatures_from_code(self, code: str) -> set[str]:
        # Extract CUDA kernel signatures from a given text
        cuda_pattern = r"__global__\s+void\s+[a-zA-Z_]\w*\s*\([^)]*\)"
        results = re.findall(cuda_pattern, code)
        return set(results)

    def diff_kernel_signatures(self, original: list, optimized: list) -> dict:
        # Find the shared and new kernel signatures
        shared = original.intersection(optimized)
        new = optimized.difference(original)

        return {"shared": list(shared), "new": list(new)}

    def get_diff(self) -> dict[str, list[str]]:
        originals = self.extract_kernel_signatures_from_code(self.original_code)
        news = self.extract_kernel_signatures_from_code(self.optimized_code)

        return self.diff_kernel_signatures(originals, news)


class CudaCodeRewriter:
    """
    A class to rewrite CUDA code based on the output of the OpenAI language model.

    It performs the following task in the steps below:
    1. Extract CUDA code from the markdown text
    2. Get the kernel signatures from the original and optimized code
    3. Swap the CUDA kernels in the original code with the optimized ones based on the shared signatures
    4. For any new signatures, add the optimized code to the response
    5. If the optimized code contains any new imports or global variables, add them to the the top of the response
    """

    def __init__(self, original_code: str, llm_response: str):
        self.original = original_code

        self.optimized = CudaMarkdownarser.parse(llm_response)
        self.kernel_diff = SignatureDiff(original_code, self.optimized).get_diff()

        self.runner = GPTRunner(model=LIGHTWEIGHT_OPENAI_MODEL)
        self.prompt_builder = PromptBuilder()

    def rewrite(self):
        orig, optim = self.original, self.optimized
        messages = self.prompt_builder.build_rewrite_messages(
            orig, optim, self.kernel_diff
        )
        try:
            return self.runner.get_gpt_response_from_messages(messages)
        except Exception as e:
            raise ValueError(f"Error requesting GPT response: {str(e)}")


if __name__ == "__main__":
    new_code = """__global__ void gpuMM(float *A, float *B, float *C, int N)
{
    // Optimized Matrix multiplication for NxN matrices C=A*B using shared memory
    // Each thread computes a single element of C
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;
    int numBlocks = N / BLOCK_SIZE;

    for (int block = 0; block < numBlocks; ++block) {
        // Load A and B matrices into shared memory
        As[threadIdx.y][threadIdx.x] = A[row * N + (block * BLOCK_SIZE + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(block * BLOCK_SIZE + threadIdx.y) * N + col];
        __syncthreads(); // Ensure all threads have loaded their parts before computation

        // Compute partial product for the block
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads(); // Ensure all threads have finished computation before loading next block
    }

    // Write the computed element to the matrix C
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}"""

    original_code = """
// source: https://github.com/emandere/CudaProject/blob/master/cudamatrix.cu

#include<ctime>
#include<iostream>
using namespace std;

#define BLOCK_SIZE 32

__global__ void gpuMM(float *A, float *B, float *C, int N)
{
	// Matrix multiplication for NxN matrices C=A*B
	// Each thread computes a single element of C
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	float sum = 0.f;
	for (int n = 0; n < N; ++n)
	    sum += A[row*N+n]*B[n*N+col];

	C[row*N+col] = sum;
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
	long size = N*N*sizeof(float);	// Size of the memory in bytes
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
"""
    rewriter = CudaCodeRewriter(original_code, new_code)
    print(rewriter.rewrite())
