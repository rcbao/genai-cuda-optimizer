#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define G 6.67430e-11  // Gravitational constant

struct Body {
    float x, y, z;    // Position
    float vx, vy, vz; // Velocity
    float mass;
};

// Kernel to compute forces and update velocities
__global__ void updateVelocities(Body *bodies, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float fx = 0, fy = 0, fz = 0;
        for (int j = 0; j < n; j++) {
            if (i != j) {
                float dx = bodies[j].x - bodies[i].x;
                float dy = bodies[j].y - bodies[i].y;
                float dz = bodies[j].z - bodies[i].z;
                float distSqr = dx * dx + dy * dy + dz * dz + 1e-10f; // Adding small constant to avoid division by zero
                float invDist = 1.0f / sqrt(distSqr);
                float invDist3 = invDist * invDist * invDist;
                float force = G * bodies[i].mass * bodies[j].mass * invDist3;
                fx += force * dx;
                fy += force * dy;
                fz += force * dz;
            }
        }
        // Update velocities based on computed force
        bodies[i].vx += fx / bodies[i].mass;
        bodies[i].vy += fy / bodies[i].mass;
        bodies[i].vz += fz / bodies[i].mass;
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

    // Launch the kernel
    updateVelocities<<<blocksPerGrid, threadsPerBlock>>>(d_bodies, numBodies);

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
