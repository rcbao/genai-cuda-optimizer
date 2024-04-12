module load cuda-toolkit-11.8.0
nvcc sample_01_matrix_mul.cu -o sample_01_matrix_mul
nvcc sample_02_vector_add.cu -o sample_02_vector_add
nvcc sample_03_nbody.cu -o sample_03_nbody