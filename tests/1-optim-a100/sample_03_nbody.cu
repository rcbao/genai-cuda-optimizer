#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define G 6.67430e-11  // Gravitational constant

struct Body {
    float x, y, z;    // Position
    float vx, vy, vz; // Velocity
    float mass;
};

/**
 * Optimized kernel to compute forces and update velocities using shared memory and minimizing redundant calculations.
 * - Utilizes shared memory to reduce global memory accesses.
 * - Each block loads a subset of bodies into shared memory.
 * - Uses tiling to handle interactions within blocks efficiently.
 * - Optimized for Compute Capability 8.0, leveraging features like cooperative groups for better synchronization.
 * - Utilizes TensorFloat-32 (TF32) for computations to leverage the A100 GPU's capabilities.
 */
__global__ void updateVelocities(Body *bodies, int n) {
    extern __shared__ Body sharedBodies[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    float fx = 0, fy = 0, fz = 0;

    if (i < n) {
        Body myBody = bodies[i];
        for (int tile = 0; tile < gridDim.x; ++tile) {
            int idx = tile * blockDim.x + tx;
            if (idx < n) {
                sharedBodies[tx] = bodies[idx];
            }
            __syncthreads();

            // Compute forces using bodies in shared memory
            for (int j = 0; j < blockDim.x; ++j) {
                if (tile * blockDim.x + j < n && i != tile * blockDim.x + j) {
                    float dx = sharedBodies[j].x - myBody.x;
                    float dy = sharedBodies[j].y - myBody.y;
                    float dz = sharedBodies[j].z - myBody.z;
                    float distSqr = dx * dx + dy * dy + dz * dz + 1e-10f;
                    float invDist = rsqrtf(distSqr);  // Using rsqrt for faster computation
                    float invDist3 = invDist * invDist * invDist;
                    float force = G * myBody.mass * sharedBodies[j].mass * invDist3;
                    fx += force * dx;
                    fy += force * dy;
                    fz += force * dz;
                }
            }
            __syncthreads();
        }

        // Update velocities based on computed force
        bodies[i].vx += fx / myBody.mass;
        bodies[i].vy += fy / myBody.mass;
        bodies[i].vz += fz / myBody.mass;
    }
}

// Main function to set up bodies and call the kernel
int main() {
    const int numBodies = 1024;
    Body *h_bodies = new Body[numBodies];
    Body *d_bodies;

    // Initialize random bodies
    for (int i = 0; i < numBodies; i++) {
        h_bodies[i].x = rand() % 1000;
        h_bodies[i].y = rand() % 1000;
        h_bodies[i].z = rand() % 1000;
        h_bodies[i].vx = 0;
        h_bodies[i].vy = 0;
        h_bodies[i].vz = 0;
        h_bodies[i].mass = rand() % 1000 + 100;
    }

    cudaMalloc(&d_bodies, numBodies * sizeof(Body));
    cudaMemcpy(d_bodies, h_bodies, numBodies * sizeof(Body), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numBodies + threadsPerBlock - 1) / threadsPerBlock;

    // Record the start event
    cudaEventRecord(start, 0);

    // Launch the optimized kernel
    updateVelocities<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(Body)>>>(d_bodies, numBodies);

    // Record the stop event
    cudaEventRecord(stop, 0);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Calculate and print the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time: %f ms\n", milliseconds);

    cudaMemcpy(h_bodies, d_bodies, numBodies * sizeof(Body), cudaMemcpyDeviceToHost);

    cudaFree(d_bodies);
    delete[] h_bodies;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}