# OpenMP vs CUDA for Otsu Thresholding

This project implements and compares Otsu Thresholding and binary image segmentation using three approaches:

- Sequential C++ baseline
- OpenMP CPU parallel implementation
- CUDA GPU implementation

The benchmark compares histogram computation, Otsu threshold determination, binary threshold application, total execution time, speedup, and the impact of image resolution.

## Implemented Features

- PGM P5 grayscale image loading and saving
- Synthetic grayscale dataset generation
- Sequential histogram, Otsu threshold, and binary segmentation
- OpenMP histogram and thresholding stages
- CUDA histogram and thresholding kernels
- Correctness validation against the sequential baseline
- CSV benchmark output in `results/`
- Graph generation in `report/graphs/`

## Requirements

- CMake 3.20 or newer
- C++17 compiler
- OpenMP-capable compiler
- NVIDIA CUDA Toolkit
- NVIDIA GPU with CUDA support
- Python 3
- Python packages: `pandas`, `matplotlib`

## Build

Configure the project with CMake, then build the Release executable:

```powershell
cmake -S . -B build
cmake --build build --config Release
```

The benchmark executable is created at:

```text
build/Release/otsu_benchmark.exe
```

## Run

```powershell
.\build\Release\otsu_benchmark.exe
```

The program processes all PGM images in `data/`, validates OpenMP and CUDA results against the sequential implementation, writes segmented images to `output/`, and writes timings to `results/results.csv`.

## Dataset Generation

Generate the synthetic benchmark dataset:

```powershell
python scripts\generate_dataset.py
```

Generate a single small test PGM image:

```powershell
python scripts\generate_test_pgm.py
```

The project uses binary PGM P5 grayscale images to avoid external image-processing dependencies such as OpenCV.

## Benchmarks

Run Release benchmarks with different OpenMP thread counts:

```powershell
cmake -E env OMP_NUM_THREADS=1 .\build\Release\otsu_benchmark.exe
Copy-Item results\results.csv results\results_release_threads_1.csv

cmake -E env OMP_NUM_THREADS=2 .\build\Release\otsu_benchmark.exe
Copy-Item results\results.csv results\results_release_threads_2.csv

cmake -E env OMP_NUM_THREADS=4 .\build\Release\otsu_benchmark.exe
Copy-Item results\results.csv results\results_release_threads_4.csv

cmake -E env OMP_NUM_THREADS=8 .\build\Release\otsu_benchmark.exe
Copy-Item results\results.csv results\results_release_threads_8.csv

cmake -E env OMP_NUM_THREADS=16 .\build\Release\otsu_benchmark.exe
Copy-Item results\results.csv results\results_release_threads_16.csv
```

## Graph Generation

Generate report graphs from the Release CSV files:

```powershell
python scripts\generate_graphs.py
```

The generated graphs are saved in:

```text
report/graphs/
```

## Output Folders

- `data/`: input PGM images
- `output/sequential/`: segmented images from the sequential implementation
- `output/openmp/`: segmented images from the OpenMP implementation
- `output/cuda/`: segmented images from the CUDA implementation
- `results/`: benchmark CSV files
- `report/`: written report
- `report/graphs/`: generated performance graphs

## Git Branches

- `main`: base project branch
- `sequential-implementation`: sequential C++ implementation
- `openmp-implementation`: OpenMP implementation
- `cuda-implementation`: CUDA implementation
- `benchmarking-and-report`: final benchmark results, graphs, README, and report
