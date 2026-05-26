#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <stdexcept>
#include <filesystem>
#include <iomanip>
#include <omp.h>

namespace fs = std::filesystem;

struct Image {
    int width;
    int height;
    std::vector<unsigned char> pixels;
};

struct BenchmarkResult {
    std::string imageName;
    int width;
    int height;
    int totalPixels;
    std::string method;
    int threads;
    double histogramTimeMs;
    double otsuTimeMs;
    double thresholdingTimeMs;
    double totalTimeMs;
    double speedup;
    int otsuThreshold;
};

// ---------------------------------------------------------
// Load PGM image - supports P5 binary grayscale PGM
// ---------------------------------------------------------
Image loadPGM(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open input file: " + filePath);
    }

    std::string magicNumber;
    file >> magicNumber;

    if (magicNumber != "P5") {
        throw std::runtime_error("Only P5 binary PGM files are supported.");
    }

    int width;
    int height;
    int maxValue;

    file >> width >> height >> maxValue;

    if (maxValue != 255) {
        throw std::runtime_error("Only PGM files with max value 255 are supported.");
    }

    file.get();

    std::vector<unsigned char> pixels(width * height);
    file.read(reinterpret_cast<char*>(pixels.data()), pixels.size());

    if (!file) {
        throw std::runtime_error("Error while reading pixel data from: " + filePath);
    }

    return Image{width, height, pixels};
}

// ---------------------------------------------------------
// Save PGM image
// ---------------------------------------------------------
void savePGM(const std::string& filePath, const Image& image) {
    std::ofstream file(filePath, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open output file: " + filePath);
    }

    file << "P5\n";
    file << image.width << " " << image.height << "\n";
    file << "255\n";

    file.write(reinterpret_cast<const char*>(image.pixels.data()), image.pixels.size());
}

// ---------------------------------------------------------
// Sequential histogram
// ---------------------------------------------------------
std::vector<unsigned int> computeHistogramSequential(
    const std::vector<unsigned char>& image
) {
    std::vector<unsigned int> histogram(256, 0);

    for (size_t i = 0; i < image.size(); i++) {
        unsigned char pixel = image[i];
        histogram[pixel]++;
    }

    return histogram;
}

// ---------------------------------------------------------
// OpenMP histogram
// ---------------------------------------------------------
std::vector<unsigned int> computeHistogramOpenMP(
    const std::vector<unsigned char>& image
) {
    int numberOfThreads = omp_get_max_threads();

    std::vector<std::vector<unsigned int>> localHistograms(
        numberOfThreads,
        std::vector<unsigned int>(256, 0)
    );

    #pragma omp parallel
    {
        int threadId = omp_get_thread_num();

        #pragma omp for
        for (long long i = 0; i < static_cast<long long>(image.size()); i++) {
            unsigned char pixel = image[i];
            localHistograms[threadId][pixel]++;
        }
    }

    std::vector<unsigned int> finalHistogram(256, 0);

    for (int thread = 0; thread < numberOfThreads; thread++) {
        for (int value = 0; value < 256; value++) {
            finalHistogram[value] += localHistograms[thread][value];
        }
    }

    return finalHistogram;
}

// ---------------------------------------------------------
// Otsu threshold calculation
// ---------------------------------------------------------
int computeOtsuThreshold(
    const std::vector<unsigned int>& histogram,
    int totalPixels
) {
    double sumTotal = 0.0;

    for (int intensity = 0; intensity < 256; intensity++) {
        sumTotal += intensity * histogram[intensity];
    }

    double sumBackground = 0.0;
    int weightBackground = 0;
    int weightForeground = 0;

    double maxBetweenClassVariance = 0.0;
    int bestThreshold = 0;

    for (int threshold = 0; threshold < 256; threshold++) {
        weightBackground += histogram[threshold];

        if (weightBackground == 0) {
            continue;
        }

        weightForeground = totalPixels - weightBackground;

        if (weightForeground == 0) {
            break;
        }

        sumBackground += threshold * histogram[threshold];

        double meanBackground = sumBackground / weightBackground;
        double meanForeground = (sumTotal - sumBackground) / weightForeground;

        double difference = meanBackground - meanForeground;

        double betweenClassVariance =
            weightBackground * weightForeground * difference * difference;

        if (betweenClassVariance > maxBetweenClassVariance) {
            maxBetweenClassVariance = betweenClassVariance;
            bestThreshold = threshold;
        }
    }

    return bestThreshold;
}

// ---------------------------------------------------------
// Sequential thresholding
// ---------------------------------------------------------
std::vector<unsigned char> applyThresholdSequential(
    const std::vector<unsigned char>& image,
    int threshold
) {
    std::vector<unsigned char> output(image.size());

    for (size_t i = 0; i < image.size(); i++) {
        output[i] = image[i] > threshold ? 255 : 0;
    }

    return output;
}

// ---------------------------------------------------------
// OpenMP thresholding
// ---------------------------------------------------------
std::vector<unsigned char> applyThresholdOpenMP(
    const std::vector<unsigned char>& image,
    int threshold
) {
    std::vector<unsigned char> output(image.size());

    #pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(image.size()); i++) {
        output[i] = image[i] > threshold ? 255 : 0;
    }

    return output;
}

// ---------------------------------------------------------
// Measure execution time
// ---------------------------------------------------------
template <typename Function>
double measureTimeMs(Function functionToMeasure) {
    auto start = std::chrono::high_resolution_clock::now();

    functionToMeasure();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}

// ---------------------------------------------------------
// Compare histograms
// ---------------------------------------------------------
bool compareHistograms(
    const std::vector<unsigned int>& first,
    const std::vector<unsigned int>& second
) {
    if (first.size() != second.size()) {
        return false;
    }

    for (size_t i = 0; i < first.size(); i++) {
        if (first[i] != second[i]) {
            return false;
        }
    }

    return true;
}

// ---------------------------------------------------------
// Compare images
// ---------------------------------------------------------
bool compareImages(
    const std::vector<unsigned char>& first,
    const std::vector<unsigned char>& second
) {
    if (first.size() != second.size()) {
        return false;
    }

    for (size_t i = 0; i < first.size(); i++) {
        if (first[i] != second[i]) {
            return false;
        }
    }

    return true;
}

// ---------------------------------------------------------
// Write benchmark results to CSV
// ---------------------------------------------------------
void writeResultsToCSV(
    const std::string& filePath,
    const std::vector<BenchmarkResult>& results
) {
    std::ofstream file(filePath);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open CSV file: " + filePath);
    }

    file << "image,width,height,total_pixels,method,threads,"
         << "histogram_ms,otsu_ms,thresholding_ms,total_ms,speedup,otsu_threshold\n";

    file << std::fixed << std::setprecision(6);

    for (const BenchmarkResult& result : results) {
        file << result.imageName << ","
             << result.width << ","
             << result.height << ","
             << result.totalPixels << ","
             << result.method << ","
             << result.threads << ","
             << result.histogramTimeMs << ","
             << result.otsuTimeMs << ","
             << result.thresholdingTimeMs << ","
             << result.totalTimeMs << ","
             << result.speedup << ","
             << result.otsuThreshold << "\n";
    }
}

// ---------------------------------------------------------
// Process one image
// ---------------------------------------------------------
void processImage(
    const std::string& inputPath,
    std::vector<BenchmarkResult>& results
) {
    std::cout << "\nProcessing image: " << inputPath << std::endl;

    Image inputImage = loadPGM(inputPath);

    int totalPixels = static_cast<int>(inputImage.pixels.size());
    std::string imageName = fs::path(inputPath).filename().string();

    std::cout << "Image size: " << inputImage.width << " x " << inputImage.height << std::endl;
    std::cout << "Total pixels: " << totalPixels << std::endl;
    std::cout << "OpenMP max threads: " << omp_get_max_threads() << std::endl;

    std::vector<unsigned int> histogramSequential;
    std::vector<unsigned int> histogramOpenMP;

    int thresholdSequential = 0;
    int thresholdOpenMP = 0;

    std::vector<unsigned char> outputSequentialPixels;
    std::vector<unsigned char> outputOpenMPPixels;

    // -----------------------------
    // Sequential version
    // -----------------------------
    double sequentialHistogramTime = measureTimeMs([&]() {
        histogramSequential = computeHistogramSequential(inputImage.pixels);
    });

    double sequentialOtsuTime = measureTimeMs([&]() {
        thresholdSequential = computeOtsuThreshold(histogramSequential, totalPixels);
    });

    double sequentialThresholdingTime = measureTimeMs([&]() {
        outputSequentialPixels = applyThresholdSequential(inputImage.pixels, thresholdSequential);
    });

    double sequentialTotalTime =
        sequentialHistogramTime +
        sequentialOtsuTime +
        sequentialThresholdingTime;

    // -----------------------------
    // OpenMP version
    // -----------------------------
    double openmpHistogramTime = measureTimeMs([&]() {
        histogramOpenMP = computeHistogramOpenMP(inputImage.pixels);
    });

    double openmpOtsuTime = measureTimeMs([&]() {
        thresholdOpenMP = computeOtsuThreshold(histogramOpenMP, totalPixels);
    });

    double openmpThresholdingTime = measureTimeMs([&]() {
        outputOpenMPPixels = applyThresholdOpenMP(inputImage.pixels, thresholdOpenMP);
    });

    double openmpTotalTime =
        openmpHistogramTime +
        openmpOtsuTime +
        openmpThresholdingTime;

    // -----------------------------
    // Validation
    // -----------------------------
    bool sameHistogram = compareHistograms(histogramSequential, histogramOpenMP);
    bool sameThreshold = thresholdSequential == thresholdOpenMP;
    bool sameOutput = compareImages(outputSequentialPixels, outputOpenMPPixels);

    std::cout << "Sequential total time: " << sequentialTotalTime << " ms" << std::endl;
    std::cout << "OpenMP total time:     " << openmpTotalTime << " ms" << std::endl;

    if (openmpTotalTime > 0) {
        std::cout << "Speedup: " << sequentialTotalTime / openmpTotalTime << "x" << std::endl;
    }

    std::cout << "Sequential Otsu threshold: " << thresholdSequential << std::endl;
    std::cout << "OpenMP Otsu threshold:     " << thresholdOpenMP << std::endl;

    std::cout << "Same histogram: " << (sameHistogram ? "YES" : "NO") << std::endl;
    std::cout << "Same threshold: " << (sameThreshold ? "YES" : "NO") << std::endl;
    std::cout << "Same output:    " << (sameOutput ? "YES" : "NO") << std::endl;

    if (!sameHistogram || !sameThreshold || !sameOutput) {
        throw std::runtime_error("Validation failed for image: " + imageName);
    }

    // -----------------------------
    // Save output images
    // -----------------------------
    fs::create_directories("output/sequential");
    fs::create_directories("output/openmp");

    std::string sequentialOutputPath =
        "output/sequential/" + fs::path(inputPath).stem().string() + "_sequential.pgm";

    std::string openmpOutputPath =
        "output/openmp/" + fs::path(inputPath).stem().string() + "_openmp.pgm";

    Image sequentialOutputImage{
        inputImage.width,
        inputImage.height,
        outputSequentialPixels
    };

    Image openmpOutputImage{
        inputImage.width,
        inputImage.height,
        outputOpenMPPixels
    };

    savePGM(sequentialOutputPath, sequentialOutputImage);
    savePGM(openmpOutputPath, openmpOutputImage);

    // -----------------------------
    // Store benchmark results
    // -----------------------------
    results.push_back(BenchmarkResult{
        imageName,
        inputImage.width,
        inputImage.height,
        totalPixels,
        "sequential",
        1,
        sequentialHistogramTime,
        sequentialOtsuTime,
        sequentialThresholdingTime,
        sequentialTotalTime,
        1.0,
        thresholdSequential
    });

    double speedup = 0.0;

    if (openmpTotalTime > 0) {
        speedup = sequentialTotalTime / openmpTotalTime;
    }

    results.push_back(BenchmarkResult{
        imageName,
        inputImage.width,
        inputImage.height,
        totalPixels,
        "openmp",
        omp_get_max_threads(),
        openmpHistogramTime,
        openmpOtsuTime,
        openmpThresholdingTime,
        openmpTotalTime,
        speedup,
        thresholdOpenMP
    });
}

// ---------------------------------------------------------
// Main
// ---------------------------------------------------------
int main() {
    try {
        std::cout << "Otsu OpenMP benchmark started!" << std::endl;

        fs::create_directories("results");

        std::vector<std::string> inputImages = {
            "data/test_512x512.pgm",
            "data/test_1024x1024.pgm",
            "data/test_2048x2048.pgm",
            "data/test_3840x2160.pgm"
        };

        std::vector<BenchmarkResult> results;

        for (const std::string& inputPath : inputImages) {
            processImage(inputPath, results);
        }

        std::string csvPath = "results/results.csv";
        writeResultsToCSV(csvPath, results);

        std::cout << "\nBenchmark completed successfully." << std::endl;
        std::cout << "CSV saved to: " << csvPath << std::endl;

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }
}
