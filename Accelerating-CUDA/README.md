# Accelerating Applications with CUDA C/C++

These are my solutions for the exercises from the 2023 version of Accelerating Applications With CUDA chapter of the NVIDIA's [Fundamentals of Accelerated Computing with CUDA C/C++](https://courses.nvidia.com/courses/course-v1:DLI+C-AC-01+V1/about) course.

## Dependences

You should have CMake and the NVIDIA CUDA Toolkit installed.

## Building

Even though the course's Jupyter notebook provides how to call the `nvcc` compiler, a CMakeLists file is provided to automatize the compilation process.

1. Make a `/build` directory and change into it
1. Generate the build files with `cmake ..`
1. Compile with `cmake --build .`
