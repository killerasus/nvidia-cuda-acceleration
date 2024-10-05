cmake_minimum_required(VERSION 3.21)

project(AcceleratingWithCUDA 
        DESCRIPTION "Solutions for the exercises of the NVIDIA course Fundamentals of Accelerated Computing with CUDA/C++."
        LANGUAGES CUDA
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(01-HelloWorld)
add_subdirectory(02-FirstParallel)
add_subdirectory(03-ThreadandBlockIndexes)
add_subdirectory(04-SingleBlockLoop)
add_subdirectory(05-MultiBlockLoop)
add_subdirectory(06-DoubleElements)
add_subdirectory(07-MismatchedConfigLoop)
add_subdirectory(08-GridStrideDouble)
add_subdirectory(09-AddErrorHandling)
add_subdirectory(10-VectorAdd)
add_subdirectory(11-Extra-MatrixMultiply)
add_subdirectory(12-Extra-AccelerateAThermalConductivityApplication)
