#include <iostream>

extern "C" void testCudaFromCpp() {
    std::cout << "CUDA file is linked successfully!" << std::endl;
}