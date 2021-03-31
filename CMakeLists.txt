cmake_minimum_required(VERSION 3.17)
set(CMAKE_CUDA_COMPILER "/usr/bin/cuda/bin/nvcc")
project(imageProcessingCuda CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(imageProcessingCuda main.cu)

set_target_properties(
        imageProcessingCuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)