/* MIT License
 *
 * Copyright (c) 2019 - Folke Vesterlund
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <map>

#include "CCL.cuh"
#include "cudaCCL.cuh"
#include "utils.hpp"
#include "timer.h"

#include <npp.h>
#include <nppdefs.h>
#include <nppcore.h>
#include <nppi_filtering_functions.h>

__global__ void extractBoundingBoxes(const unsigned int* labels, int numRows, int numCols, int* boundingBoxes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numRows * numCols) {
        unsigned int label = labels[tid];
        atomicMin(&boundingBoxes[4 * label], tid % numCols);    // min x
        atomicMax(&boundingBoxes[4 * label + 1], tid % numCols); // max x
        atomicMin(&boundingBoxes[4 * label + 2], tid / numCols); // min y
        atomicMax(&boundingBoxes[4 * label + 3], tid / numCols); // max y
    }
}


__global__ void computeMean(const unsigned char* image, int numRows, int numCols, int* mean) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    unsigned long long sum = 0;
    int count = 0;

    for (int i = tid; i < numRows * numCols; i += stride) {
        sum += image[i];
        count++;
    }

    // Use atomicAdd to safely update mean value from multiple threads
    atomicAdd(mean, sum);
    atomicAdd(mean + 1, count);
}

int getMean(const unsigned char* d_image, int numRows, int numCols) {
    int numPixels = numRows * numCols;
    int* d_mean;
    cudaMalloc(&d_mean, 2 * sizeof(int));
    cudaMemset(d_mean, 0, 2 * sizeof(int));

    // Set grid and block dimensions
    int threadsPerBlock = 256;
    int numBlocks = (numPixels + threadsPerBlock - 1) / threadsPerBlock;

    // Run the kernel
    computeMean<<<numBlocks, threadsPerBlock>>>(d_image, numRows, numCols, d_mean);

    // Copy mean result from device to host
    int h_mean[2];
    cudaMemcpy(h_mean, d_mean, 2 * sizeof(int), cudaMemcpyDeviceToHost);

    // Compute the mean and return it as an integer
    int meanValue = (h_mean[0] + h_mean[1] / 2) / h_mean[1];

    // Free device memory
    cudaFree(d_mean);

    return meanValue;
}


__global__ void thresholdImage(unsigned char* image, int numRows, int numCols, int threshold) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < numRows * numCols; i += stride) {
        image[i] = (image[i] > threshold) ? 255 : 0;
    }
}

void thresholdImageInPlace(unsigned char* d_image, int numRows, int numCols, int threshold) {
    // Set grid and block dimensions
    int threadsPerBlock = 256;
    int numBlocks = (numRows * numCols + threadsPerBlock - 1) / threadsPerBlock;

    // Run the kernel
    thresholdImage<<<numBlocks, threadsPerBlock>>>(d_image, numRows, numCols, threshold);
}


__global__ void initBoundingBoxes(int* boundingBoxes, int numBoundingBoxes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < numBoundingBoxes; i += stride) {
        boundingBoxes[4 * i] = INT_MAX;  // Initialize x with maximum value
        boundingBoxes[4 * i + 1] = 0;    // Initialize y with 0
        boundingBoxes[4 * i + 2] = INT_MAX;  // Initialize width with maximum value
        boundingBoxes[4 * i + 3] = 0;    // Initialize height with 0
    }
}

void initializeBoundingBoxes(int* d_boundingBoxes, int numBoundingBoxes) {
    // Set grid and block dimensions
    int threadsPerBlock = 256;
    int numBlocks = (numBoundingBoxes + threadsPerBlock - 1) / threadsPerBlock;

    // Run the kernel
    initBoundingBoxes<<<numBlocks, threadsPerBlock>>>(d_boundingBoxes, numBoundingBoxes);
}


void processCCL(char* d_img, int numRows, int numCols, int* numComponents, int* d_boundingBoxes) {
    int numPixels = numRows * numCols;
    
    unsigned int* d_labels;
    cudaMallocManaged(&d_labels, numPixels * sizeof(unsigned int));
    // cudaMalloc(&d_img, numPixels * sizeof(unsigned char));

    // Pre-process image
    // cudaMemcpy(d_img, image, numPixels * sizeof(unsigned char), cudaMemcpyHostToDevice);


    // Pre process image
	unsigned int imgMean = getMean((unsigned char * )d_img, numRows,numCols);
	thresholdImageInPlace((unsigned char *)d_img, numRows, numCols,imgMean);


    // Run and time kernel
    GpuTimer timer;
    timer.Start();
    connectedComponentLabeling(d_labels, (unsigned char *)d_img, numCols, numRows);
    timer.Stop();
    std::cout << "\nGPU code ran in: " << timer.Elapsed() << "ms" << std::endl;
    

    timer.Start();
    // Compress the sparse labels using NPP functions
    int bufferSize;
    NppStatus status = nppiCompressMarkerLabelsGetBufferSize_32u_C1R(numRows * numCols, &bufferSize);
    if (status != NPP_SUCCESS) {
        // Handle NPP error
    }

    unsigned char* d_buffer;
    cudaMalloc(&d_buffer, bufferSize * sizeof(unsigned char));

    int newNumber;
    status = nppiCompressMarkerLabelsUF_32u_C1IR(d_labels, numCols * sizeof(unsigned int),{numCols, numRows}, numRows * numCols, &newNumber, d_buffer);
    if (status != NPP_SUCCESS) {
        // Handle NPP error
    }

    std::cout<<"components : " << newNumber << std::endl;

    // // Allocate memory for bounding boxes on the host
    // int* h_boundingBoxes = new int[newNumber * 4];
    // for (int i = 0; i < newNumber * 4; i+=2) {
    //     h_boundingBoxes[i] = INT_MAX;  // Initialize with maximum values
    // }
    // for (int i = 1; i < newNumber * 4; i+=2) {
    //     h_boundingBoxes[i] = 0;  // Initialize with maximum values
    // }

    initializeBoundingBoxes(d_boundingBoxes,newNumber);


    // // Allocate memory for bounding boxes on the device
    // int* d_boundingBoxes;
    // cudaMalloc(&d_boundingBoxes, newNumber * 4 * sizeof(int));

    // // Copy bounding boxes memory from host to device
    // cudaMemcpy(d_boundingBoxes, h_boundingBoxes, newNumber * 4 * sizeof(int), cudaMemcpyHostToDevice);

    // Set grid and block dimensions
    int threadsPerBlock = 256;
    int numBlocks = (numPixels + threadsPerBlock - 1) / threadsPerBlock;

    // Extract bounding boxes on the GPU
    extractBoundingBoxes<<<numBlocks, threadsPerBlock>>>(d_labels, numRows, numCols, d_boundingBoxes);

    // Copy bounding boxes memory from device to host
    // cudaMemcpy(h_boundingBoxes, d_boundingBoxes, newNumber * 4 * sizeof(int), cudaMemcpyDeviceToHost);

    timer.Stop();
    std::cout << "label compression, bounding box code ran in: " << timer.Elapsed() << "ms" << std::endl;

    // // Print bounding boxes
    // std::cout << "Bounding Boxes:\n";
    // for (int i = 0; i < newNumber; i++) {
    //     int minX = h_boundingBoxes[4 * i];
    //     int maxX = h_boundingBoxes[4 * i + 1];
    //     int minY = h_boundingBoxes[4 * i + 2];
    //     int maxY = h_boundingBoxes[4 * i + 3];

    //     std::cout << "Component " << i + 1 << ": "
    //               << "MinX=" << minX << ", MaxX=" << maxX
    //               << ", MinY=" << minY << ", MaxY=" << maxY << "\n";
    // }

    // Update numComponents with the total number of components
    *numComponents = newNumber;

    // Copy bounding boxes from h_boundingBoxes to the provided boundingBoxes array
    // memcpy(boundingBoxes, h_boundingBoxes, newNumber * 4 * sizeof(int));

    // Free host and device memory
    // delete[] h_boundingBoxes;
    // cudaFree(d_img);
    cudaFree(d_labels);
    cudaFree(d_buffer);
    // cudaFree(d_boundingBoxes);
}

