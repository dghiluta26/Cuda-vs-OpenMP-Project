#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <sstream>
#include <stdexcept>

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t error = (call);                                              \
        if (error != cudaSuccess) {                                              \
            std::ostringstream message;                                          \
            message << "CUDA error at " << __FILE__ << ":" << __LINE__          \
                    << " - " << cudaGetErrorString(error);                      \
            throw std::runtime_error(message.str());                            \
        }                                                                        \
    } while (0)

// ---------------------------------------------------------
// CUDA kernel: compute histogram using shared memory
// ---------------------------------------------------------
__global__ void histogramKernel(
    const unsigned char* image,
    int size,
    unsigned int* globalHistogram
) {
    __shared__ unsigned int localHistogram[256];

    int threadId = threadIdx.x;

    if (threadId < 256) {
        localHistogram[threadId] = 0;
    }

    __syncthreads();

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (index < size) {
        unsigned char pixel = image[index];
        atomicAdd(&localHistogram[pixel], 1);
        index += stride;
    }

    __syncthreads();

    if (threadId < 256) {
        atomicAdd(&globalHistogram[threadId], localHistogram[threadId]);
    }
}

// ---------------------------------------------------------
// CUDA kernel: apply binary thresholding
// ---------------------------------------------------------
__global__ void thresholdKernel(
    const unsigned char* input,
    unsigned char* output,
    int size,
    int threshold
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (index < size) {
        output[index] = input[index] > threshold ? 255 : 0;
        index += stride;
    }
}

// ---------------------------------------------------------
// C++ callable wrapper: CUDA histogram
// ---------------------------------------------------------
extern "C" void computeHistogramCUDA(
    const unsigned char* hostImage,
    int size,
    unsigned int* hostHistogram,
    float* kernelTimeMs,
    float* totalTimeMs
) {
    if (hostImage == nullptr || hostHistogram == nullptr || size <= 0) {
        throw std::runtime_error("Invalid input passed to computeHistogramCUDA.");
    }

    *kernelTimeMs = 0.0f;
    *totalTimeMs = 0.0f;

    unsigned char* deviceImage = nullptr;
    unsigned int* deviceHistogram = nullptr;

    cudaEvent_t kernelStart;
    cudaEvent_t kernelStop;

    auto totalStart = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMalloc(&deviceImage, size * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&deviceHistogram, 256 * sizeof(unsigned int)));

    CUDA_CHECK(cudaMemcpy(
        deviceImage,
        hostImage,
        size * sizeof(unsigned char),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemset(deviceHistogram, 0, 256 * sizeof(unsigned int)));

    CUDA_CHECK(cudaEventCreate(&kernelStart));
    CUDA_CHECK(cudaEventCreate(&kernelStop));

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    gridSize = std::max(1, std::min(gridSize, 1024));

    CUDA_CHECK(cudaEventRecord(kernelStart));

    histogramKernel<<<gridSize, blockSize>>>(
        deviceImage,
        size,
        deviceHistogram
    );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(kernelStop));
    CUDA_CHECK(cudaEventSynchronize(kernelStop));
    CUDA_CHECK(cudaEventElapsedTime(kernelTimeMs, kernelStart, kernelStop));

    CUDA_CHECK(cudaMemcpy(
        hostHistogram,
        deviceHistogram,
        256 * sizeof(unsigned int),
        cudaMemcpyDeviceToHost
    ));

    auto totalStop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> totalDuration = totalStop - totalStart;
    *totalTimeMs = totalDuration.count();

    CUDA_CHECK(cudaEventDestroy(kernelStart));
    CUDA_CHECK(cudaEventDestroy(kernelStop));

    CUDA_CHECK(cudaFree(deviceImage));
    CUDA_CHECK(cudaFree(deviceHistogram));
}

// ---------------------------------------------------------
// C++ callable wrapper: CUDA thresholding
// ---------------------------------------------------------
extern "C" void applyThresholdCUDA(
    const unsigned char* hostInput,
    unsigned char* hostOutput,
    int size,
    int threshold,
    float* kernelTimeMs,
    float* totalTimeMs
) {
    if (hostInput == nullptr || hostOutput == nullptr || size <= 0) {
        throw std::runtime_error("Invalid input passed to applyThresholdCUDA.");
    }

    *kernelTimeMs = 0.0f;
    *totalTimeMs = 0.0f;

    unsigned char* deviceInput = nullptr;
    unsigned char* deviceOutput = nullptr;

    cudaEvent_t kernelStart;
    cudaEvent_t kernelStop;

    auto totalStart = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMalloc(&deviceInput, size * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&deviceOutput, size * sizeof(unsigned char)));

    CUDA_CHECK(cudaMemcpy(
        deviceInput,
        hostInput,
        size * sizeof(unsigned char),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaEventCreate(&kernelStart));
    CUDA_CHECK(cudaEventCreate(&kernelStop));

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    gridSize = std::max(1, std::min(gridSize, 1024));

    CUDA_CHECK(cudaEventRecord(kernelStart));

    thresholdKernel<<<gridSize, blockSize>>>(
        deviceInput,
        deviceOutput,
        size,
        threshold
    );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(kernelStop));
    CUDA_CHECK(cudaEventSynchronize(kernelStop));
    CUDA_CHECK(cudaEventElapsedTime(kernelTimeMs, kernelStart, kernelStop));

    CUDA_CHECK(cudaMemcpy(
        hostOutput,
        deviceOutput,
        size * sizeof(unsigned char),
        cudaMemcpyDeviceToHost
    ));

    auto totalStop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> totalDuration = totalStop - totalStart;
    *totalTimeMs = totalDuration.count();

    CUDA_CHECK(cudaEventDestroy(kernelStart));
    CUDA_CHECK(cudaEventDestroy(kernelStop));

    CUDA_CHECK(cudaFree(deviceInput));
    CUDA_CHECK(cudaFree(deviceOutput));
}