cmake_minimum_required(VERSION 3.9)

project(AcceleratingWithCUDA 
        DESCRIPTION "Solutions for the exercises of the NVIDIA course Fundamentals of Accelerated Computing with CUDA/C++."
        LANGUAGES CUDA
)

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

install(TARGETS
        01-hello-gpu
        01-first-parallel
        01-thread-and-block-idx
        01-single-block-loop
        02-multi-block-loop
        01-double-elements
        02-mismatched-config-loop
        03-grid-stride-double
        01-add-error-handling
        01-vector-add
        01-matrix-multiply-2d
        01-heat-conduction
        OBJECTS DESTINATION bin/
)
