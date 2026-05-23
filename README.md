# Otsu Thresholding: OpenMP vs CUDA

This project compares sequential C++, OpenMP and CUDA implementations of Otsu Thresholding for binary image segmentation.

## Implemented Features

- PGM grayscale image loading and saving
- Sequential histogram computation
- OpenMP histogram computation
- CUDA histogram computation
- Otsu threshold calculation
- Sequential thresholding
- OpenMP thresholding
- CUDA thresholding
- CSV benchmark export
- Performance graphs

## Requirements

- Windows 11
- Visual Studio 2022 with Desktop development with C++
- CMake
- NVIDIA CUDA Toolkit
- NVIDIA GPU with CUDA support
- Python 3 for graph generation

## Build

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release