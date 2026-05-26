#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

std::vector<unsigned int> computeHistogramSequential(
    const std::vector<unsigned char>& image
) {
    std::vector<unsigned int> histogram(256, 0);

    for (size_t i = 0; i < image.size(); i++) {
        histogram[image[i]]++;
    }

    return histogram;
}

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
    int bestThreshold = 0;
    double maxBetweenClassVariance = 0.0;

    for (int threshold = 0; threshold < 256; threshold++) {
        weightBackground += histogram[threshold];

        if (weightBackground == 0) {
            continue;
        }

        int weightForeground = totalPixels - weightBackground;

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

template <typename Function>
double measureTimeMs(Function functionToMeasure) {
    auto start = std::chrono::high_resolution_clock::now();
    functionToMeasure();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}

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

    std::vector<unsigned int> histogram;
    int threshold = 0;
    std::vector<unsigned char> outputPixels;

    double histogramTime = measureTimeMs([&]() {
        histogram = computeHistogramSequential(inputImage.pixels);
    });

    double otsuTime = measureTimeMs([&]() {
        threshold = computeOtsuThreshold(histogram, totalPixels);
    });

    double thresholdingTime = measureTimeMs([&]() {
        outputPixels = applyThresholdSequential(inputImage.pixels, threshold);
    });

    double totalTime = histogramTime + otsuTime + thresholdingTime;

    std::cout << "Sequential total time: " << totalTime << " ms" << std::endl;
    std::cout << "Otsu threshold: " << threshold << std::endl;

    fs::create_directories("output/sequential");

    std::string outputPath =
        "output/sequential/" + fs::path(inputPath).stem().string() + "_sequential.pgm";

    savePGM(outputPath, Image{
        inputImage.width,
        inputImage.height,
        outputPixels
    });

    results.push_back(BenchmarkResult{
        imageName,
        inputImage.width,
        inputImage.height,
        totalPixels,
        "sequential",
        1,
        histogramTime,
        otsuTime,
        thresholdingTime,
        totalTime,
        1.0,
        threshold
    });
}

int main() {
    try {
        std::cout << "Otsu sequential benchmark started!" << std::endl;

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
