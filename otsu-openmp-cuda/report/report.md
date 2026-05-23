# OpenMP vs CUDA for Otsu Thresholding and Binary Segmentation

## 1. Introduction

This project compares CPU multi-core parallelization using OpenMP with GPU parallelization using CUDA for Otsu Thresholding and binary image segmentation.

## 2. Algorithm Description

Otsu Thresholding is an automatic threshold selection method used for grayscale image segmentation. The algorithm first computes a 256-bin grayscale histogram, then searches for the threshold that maximizes the between-class variance. After the threshold is found, each pixel is converted to either 0 or 255.

## 3. Implementations

### 3.1 Sequential Implementation

The sequential version computes the histogram and applies thresholding using standard C++ loops.

### 3.2 OpenMP Implementation

The OpenMP version parallelizes the histogram computation using local histograms for each thread. The thresholding stage is parallelized using `#pragma omp parallel for`.

### 3.3 CUDA Implementation

The CUDA version computes the histogram on the GPU using a CUDA kernel with shared memory and atomic operations. The thresholding step is also executed on the GPU using one CUDA kernel.

## 4. Experimental Setup

- CPU: Intel i7-14700F
- GPU: NVIDIA RTX 4060
- Operating System: Windows 11
- Compiler: MSVC / NVCC
- Build mode: Release
- Dataset: synthetic grayscale PGM images with multiple resolutions

## 5. Results

Include the generated graphs:

- `total_time_vs_resolution.png`
- `histogram_time_vs_resolution.png`
- `thresholding_time_vs_resolution.png`
- `openmp_speedup_vs_threads.png`
- `cuda_speedup_vs_resolution.png`

## 6. Interpretation

The OpenMP implementation improves execution time by distributing pixel-level operations across multiple CPU threads. However, the speedup is not perfectly linear due to overhead, memory bandwidth limitations and synchronization costs during histogram computation.

The CUDA implementation can be faster for large images because the GPU can process many pixels in parallel. However, CUDA total time also includes memory transfers between CPU and GPU, so for smaller images the advantage may be reduced.

The Otsu threshold computation itself is very fast because it only iterates over 256 histogram bins. Therefore, the main performance differences appear in histogram computation and thresholding.

## 7. Conclusion

The project shows that both OpenMP and CUDA can accelerate Otsu Thresholding compared to the sequential implementation. OpenMP is easier to integrate and performs well on CPU, while CUDA becomes more relevant when processing larger images or large batches of images.