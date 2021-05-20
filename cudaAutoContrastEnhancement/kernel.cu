#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define NUM_THREADS 1024
#define NUM_CHANNELS 1
#define DEBUG_IMG_IDX 20

typedef unsigned char uint8_t;

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdint.h>

cudaError_t contrastEnhancementCuda(uint8_t *img, uint8_t &min_host, uint8_t &max_host, const int size);

__global__ void minKernel(uint8_t* img, uint8_t* o_img, const int size) {

    // Shared memory for threads in the same block
    extern __shared__ uint8_t sdata_minKernel[];

    unsigned int tid = threadIdx.x;
    // TODO: Check blockDim * 2 option
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Initialize shared memory
    if (i + blockDim.x < size) {
        sdata_minKernel[tid] = img[i] < img[i + blockDim.x] ? img[i] : img[i + blockDim.x];
    }
    else {
        sdata_minKernel[tid] = img[i];
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s) {
            sdata_minKernel[tid] = sdata_minKernel[i] < sdata_minKernel[i + s] ? sdata_minKernel[i] : sdata_minKernel[i + s];
        }
        __syncthreads();
    }
    
    // Unroll the last warp
    if (tid < 32) {
        sdata_minKernel[tid] = sdata_minKernel[i] < sdata_minKernel[i + 32] ? sdata_minKernel[i] : sdata_minKernel[i + 32];
        sdata_minKernel[tid] = sdata_minKernel[i] < sdata_minKernel[i + 16] ? sdata_minKernel[i] : sdata_minKernel[i + 16];
        sdata_minKernel[tid] = sdata_minKernel[i] < sdata_minKernel[i + 8] ? sdata_minKernel[i] : sdata_minKernel[i + 8];
        sdata_minKernel[tid] = sdata_minKernel[i] < sdata_minKernel[i + 4] ? sdata_minKernel[i] : sdata_minKernel[i + 4];
        sdata_minKernel[tid] = sdata_minKernel[i] < sdata_minKernel[i + 2] ? sdata_minKernel[i] : sdata_minKernel[i + 2];
        sdata_minKernel[tid] = sdata_minKernel[i] < sdata_minKernel[i + 1] ? sdata_minKernel[i] : sdata_minKernel[i + 1];
    }

    if (tid == 0) {
        o_img[blockIdx.x] = sdata_minKernel[0];
    }
}


__global__ void maxKernel(uint8_t* img, uint8_t* o_img, const int size) {

    // Shared memory for threads in the same block
    extern __shared__ uint8_t sdata_maxKernel[];

    unsigned int tid = threadIdx.x;
    // TODO: Check blockDim * 2 option
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Initialize shared memory
    if (i + blockDim.x < size) {
        sdata_maxKernel[tid] = img[i] > img[i + blockDim.x] ? img[i] : img[i + blockDim.x];
    }
    else {
        sdata_maxKernel[tid] = img[i];
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s) {
            sdata_maxKernel[tid] = sdata_maxKernel[i] > sdata_maxKernel[i + s] ? sdata_maxKernel[i] : sdata_maxKernel[i + s];
        }
        __syncthreads();
    }

    // Unroll the last warp
    if (tid < 32) {
        sdata_maxKernel[tid] = sdata_maxKernel[i] > sdata_maxKernel[i + 32] ? sdata_maxKernel[i] : sdata_maxKernel[i + 32];
        sdata_maxKernel[tid] = sdata_maxKernel[i] > sdata_maxKernel[i + 16] ? sdata_maxKernel[i] : sdata_maxKernel[i + 16];
        sdata_maxKernel[tid] = sdata_maxKernel[i] > sdata_maxKernel[i + 8] ? sdata_maxKernel[i] : sdata_maxKernel[i + 8];
        sdata_maxKernel[tid] = sdata_maxKernel[i] > sdata_maxKernel[i + 4] ? sdata_maxKernel[i] : sdata_maxKernel[i + 4];
        sdata_maxKernel[tid] = sdata_maxKernel[i] > sdata_maxKernel[i + 2] ? sdata_maxKernel[i] : sdata_maxKernel[i + 2];
        sdata_maxKernel[tid] = sdata_maxKernel[i] > sdata_maxKernel[i + 1] ? sdata_maxKernel[i] : sdata_maxKernel[i + 1];
    }

    if (tid == 0) {
        o_img[blockIdx.x] = sdata_maxKernel[0];
    }
}


__global__ void subtractMinKernel(uint8_t* dev_img, uint8_t* dev_min, const int size) {

    // Shared min value for threads in the same block. No bank error since data is broadcasted.
    __shared__ uint8_t min_val;
        
    min_val = *dev_min;
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        dev_img[i] = dev_img[i] - min_val;
    }
    __syncthreads();
}


__global__ void scaleKernel(uint8_t* dev_img, float* dev_scale, const int size) {

    // Shared min value for threads in the same block. No bank error since data is broadcasted.
    __shared__ float scale;

    scale = *dev_scale;
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        // rintf rounds the float number to the closest integer 
        dev_img[i] = rintf(dev_img[i] * scale);
    }
    __syncthreads();
}


int main()
{
    int width; //image width
    int height; //image height
    int bpp;  //bytes per pixel if the image was RGB (not used)

    uint8_t min_host, max_host;
    
    // Load a grayscale bmp image to an unsigned integer array with its height and weight.
    //  (uint8_t is an alias for "unsigned char")
    uint8_t* image = stbi_load("./samples/640x426.bmp", &width, &height, &bpp, NUM_CHANNELS);

    // Print for sanity check
    printf("Bytes per pixel: %d \n", bpp / 3); //Image is grayscale, so bpp / 3;
    printf("Height: %d \n", height);
    printf("Width: %d \n", width);

    printf("\n### Orig val: %d \n", image[DEBUG_IMG_IDX]);

    cudaError_t cudaStatus = contrastEnhancementCuda(image, min_host, max_host, width * height);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "contrastEnhancementCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    // Write image array into a bmp file
    stbi_write_bmp("./out_img_640x426.bmp", width, height, 1, image);

    // Deallocate memory
    stbi_image_free(image);

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t contrastEnhancementCuda(uint8_t *img, uint8_t &min_host, uint8_t &max_host, const int size)
{
    int blockSize = NUM_THREADS;
    int gridSize = size / blockSize + (size % blockSize != 0);

    dim3 grid, block;
    block.x = NUM_THREADS;
    grid.x = size / block.x + (size % block.x != 0);
    
    // Temp CPU array
    uint8_t* min_array = new uint8_t [grid.x];
    //uint8_t* min_array = (uint8_t*) malloc( sizeof(uint8_t) * (size / NUM_THREADS + (size % NUM_THREADS != 0))) ;

    // Device memory pointers
    uint8_t *dev_img = 0;
    uint8_t *dev_min_array = 0;

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);

    // Allocate GPU memory for the image.
    cudaStatus = cudaMalloc((void**)&dev_img, size * sizeof(uint8_t));
    //cudaMemset(dev_img, 0, size * sizeof(uint8_t));

    // Allocate GPU memory for min values of separate blocks.
    cudaStatus = cudaMalloc((void**)&dev_min_array, sizeof(uint8_t) * grid.x);
    //cudaMemset(dev_min_array, 0, sizeof(uint8_t) * grid.x);

    // Copy the image from host memory to GPU.
    cudaMemcpy(dev_img, img, size * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMempcy img failed!");
        goto Error;
    }
    
    int smem_size = block.x * sizeof(uint8_t);
 

    //// Launch a kernel on the GPU with one thread for each element.
    //minKernel<<<grid, block, smem_size, 0>>>(dev_img, dev_min_array, size);

    //// Check for any errors launching the kernel
    //cudaStatus = cudaGetLastError();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "minMaxKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    //    goto Error;
    //}
   
    //// Copy output vector from GPU buffer to host memory.
    //cudaStatus = cudaMemcpy(min_array, dev_min_array, sizeof(uint8_t) * grid.x, cudaMemcpyDeviceToHost);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMemcpy min failed!");
    //    goto Error;
    //}
    //min_host = 255;
    //for (size_t i = 0; i < NUM_THREADS; i++)
    //{
    //    if (min_array[i] < min_host) {
    //        min_host = min_array[i];
    //    }
    //}
    //
    //maxKernel << <grid, block, smem_size, 0 >> > (dev_img, dev_min_array, size);

    //// Check for any errors launching the kernel
    //cudaStatus = cudaGetLastError();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "minMaxKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    //    goto Error;
    //}

    //// Copy output vector from GPU buffer to host memory.
    //cudaStatus = cudaMemcpy(min_array, dev_min_array, sizeof(uint8_t) * grid.x, cudaMemcpyDeviceToHost);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMemcpy max failed!");
    //    goto Error;
    //}

    //max_host = 0;
    //for (size_t i = 0; i < NUM_THREADS; i++)
    //{
    //    if (min_array[i] > max_host) {
    //        max_host = min_array[i];
    //    }
    //}
    //
    //printf("Min: %d \n Max: %d \n", min_host, max_host);
    
    // For debug purposes
    min_host= 10;
    max_host = 200;

    float scale_constant = 255.0f / (max_host - min_host);

    uint8_t* dev_min;
    float* dev_scale;

    cudaStatus = cudaMalloc((void**)&dev_min, sizeof(uint8_t));
    cudaStatus = cudaMemcpy(dev_min, &min_host, sizeof(uint8_t), cudaMemcpyHostToDevice);
    subtractMinKernel<<<grid, block, sizeof(uint8_t), 0>>> (dev_img, dev_min, size);
    cudaStatus = cudaMemcpy(img, dev_img, sizeof(uint8_t) * size, cudaMemcpyDeviceToHost);
    printf("### After subtraction %d\n", img[DEBUG_IMG_IDX]);

    cudaStatus = cudaMalloc((void**)&dev_scale, sizeof(float));
    cudaStatus = cudaMemcpy(dev_scale, &scale_constant, sizeof(float), cudaMemcpyHostToDevice);
    scaleKernel<< <grid, block, sizeof(float), 0 >> > (dev_img, dev_scale, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "scaleKernel failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(img, dev_img, sizeof(uint8_t) * size, cudaMemcpyDeviceToHost);
    printf("### After scaling %d, scale: %.3f\n", img[DEBUG_IMG_IDX], scale_constant);

Error:
    cudaFree(dev_img);
    cudaFree(dev_min_array);
    free(min_array);
    
    return cudaStatus;
}
