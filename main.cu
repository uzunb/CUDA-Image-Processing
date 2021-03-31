#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define GRIDVAL 20.0
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"


__global__ void SobelGpu(const unsigned char *gpuInputImg, unsigned char *gpuOutputImg,
                         const int width, const int height) {
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    float dx, dy;
    if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
        dx = (-1 * gpuInputImg[(y - 1) * width + (x - 1)]) + (-2 * gpuInputImg[y * width + (x - 1)]) +
             (-1 * gpuInputImg[(y + 1) * width + (x - 1)]) + (gpuInputImg[(y - 1) * width + (x + 1)]) +
             (2 * gpuInputImg[y * width + (x + 1)]) + (gpuInputImg[(y + 1) * width + (x + 1)]);
        dy = (gpuInputImg[(y - 1) * width + (x - 1)]) + (2 * gpuInputImg[(y - 1) * width + x]) +
             (gpuInputImg[(y - 1) * width + (x + 1)]) + (-1 * gpuInputImg[(y + 1) * width + (x - 1)]) +
             (-2 * gpuInputImg[(y + 1) * width + x]) + (-1 * gpuInputImg[(y + 1) * width + (x + 1)]);
        gpuOutputImg[y * width + x] = (unsigned int)sqrt((dx * dx) + (dy * dy));
    }
}


int main() {
    int width, height, channels, imageSize;
    cudaError_t result;

    char *filename = "eagle.jpg";
    unsigned char *originalImage = stbi_load(filename, &width, &height, &channels, 0);
    if (originalImage == NULL) {
        printf("Error in loading the image\n");
        exit(1);
    }
    printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);
    imageSize = width * height;


    //Memory allocation
    printf("Memory allocation\n");
    unsigned char *inputimage = (unsigned char *) malloc(imageSize * sizeof(unsigned char));
    unsigned char *outputImg = (unsigned char *) malloc(imageSize * sizeof(unsigned char));

    // Convert to Grayscale
    int imgpos = 0;
    for (int i = 0; i < imageSize; i++) {
        imgpos += channels;
        inputimage[i] = (unsigned char) (0.114 * originalImage[imgpos + 2] +
                                         0.587 * originalImage[imgpos + 1] +
                                         0.299 * originalImage[imgpos]);
    }

    // Write gray image
    filename = "eagle_gray.jpg";
    stbi_write_jpg(filename, width, height, 1, inputimage, 100);

    // GPU memory allocation
    unsigned char *gInputImage, *gOutputImage;
    result = cudaMalloc((void **) &gInputImage, imageSize * sizeof(unsigned char));
    assert(result == cudaSuccess);

    result = cudaMalloc((void **) &gOutputImage, imageSize * sizeof(unsigned char));
    assert(result == cudaSuccess);


    // Memory Transfer CPU ==> GPU
    printf("Transfering memory to gpu from cpu...\n");
    result = cudaMemcpy(gInputImage, inputimage, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
    assert(result == cudaSuccess);

    // Sobel filter
    printf("Sobel Filter executing...\n");

    dim3 threadsPerBlock(GRIDVAL, GRIDVAL, 1);
    dim3 numBlocks(ceil(width / GRIDVAL), ceil(height / GRIDVAL), 1);

    SobelGpu<<<numBlocks, threadsPerBlock>>>(gInputImage, gOutputImage, width, height);

    printf("CUDA device synchronizing...\n");
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    // Memory Transfer GPU ==> CPU
    printf("Transfering memory to cpu from gpu...\n");
    result = cudaMemcpy(outputImg, gOutputImage, imageSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    assert(result == cudaSuccess);


    printf("Output image writing...\n");
    filename = "eagle_output.jpg";
    stbi_write_jpg(filename, width, height, 1, outputImg, 100);

    // Deallocating
    stbi_image_free(originalImage);

    result = cudaFree(gOutputImage);
    assert(result == cudaSuccess);

    result = cudaFree(gInputImage);
    assert(result == cudaSuccess);

    free(outputImg);
    free(inputimage);
}
