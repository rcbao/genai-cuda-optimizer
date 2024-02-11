# compile the simple/optimized versions

module load cuda-toolkit-12.3.2

nvcc -o matrix-mul-baseline-2014-v1 matrix-mul-baseline-2014-v1.cu
nvcc -o matrix-mul-baseline-2014-v2 matrix-mul-baseline-2014-v2.cu
nvcc -o matrix-mul-baseline-2020 matrix-mul-baseline-2020.cu
nvcc -o matrix-mul-baseline-2023 matrix-mul-baseline-2023.cu

echo "Performance for matrix multiplication code (CUDA 2014, v1):"
time ./matrix-mul-baseline-2014-v1

echo "Performance for matrix multiplication code (CUDA 2014, v2):"
time ./matrix-mul-baseline-2014-v2

echo "Performance for matrix multiplication code (CUDA 2020):"
time ./matrix-mul-baseline-2020

echo "Performance for the LLM-optimized matrix multiplication:"
time ./matrix-mul-baseline-2023
