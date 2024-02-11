#!/bin/bash

# compile the simple/optimized versions
module load cuda-toolkit-12.3.2

nvcc -o matrix_mul-optimized matrix_mul-optimized.cu
nvcc -o matrix_mul matrix_mul.cu


time ./matrix_mul
time ./matrix_mul-optimized