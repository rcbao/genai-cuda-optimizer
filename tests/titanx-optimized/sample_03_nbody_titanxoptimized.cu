#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define G 6.67430e-11  // Gravitational constant

struct Body {
    float x, y, z;    // Position
    float vx, vy, vz; // Velocity
    float mass;
};

__global__ void updateVelocities(Body *bodies, int n) {
    extern __shared__ Body sharedBodies[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    Body myBody = bodies[i];

    float fx = 0, fy = 0, fz = 0;

    for (int tile = 0; tile < gridDim.x; ++tile) {
        int idx = tile * blockDim.x + tid;
        if (idx < n) {
            sharedBodies[tid] = bodies[idx];
        }
        __syncthreads();

        // Compute forces within a tile
        for (int j = 0; j < blockDim.x; ++j) {
            if (tile * blockDim.x + j < n && i != tile * blockDim.x + j) {
                float dx = sharedBodies[j].x - myBody.x;
                float dy = sharedBodies[j].y - myBody.y;
                float dz = sharedBodies[j].z - myBody.z;
                float distSqr = dx * dx + dy * dy + dz * dz + 1e-10f;
                float invDist = rsqrt(distSqr);  // Using rsqrt for faster computation
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
    if (i < n) {
        bodies[i].vx = myBody.vx + fx / myBody.mass;
        bodies[i].vy = myBody.vy + fy / myBody.mass;
        bodies[i].vz = myBody.vz + fz / myBody.mass;
    }
}

int main() {
    const int numBodies = 1024;
    Body *h_bodies = new Body[numBodies];
    Body *d_bodies;

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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numBodies + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start, 0);

    updateVelocities<<<blocksPerGrid, threadsPerBlock, numBodies * sizeof(Body)>>>(d_bodies, numBodies);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

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