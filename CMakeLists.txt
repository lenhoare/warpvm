cmake_minimum_required(VERSION 3.24)

project(WarpVM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA architecture is a build configuration, not a property of this project.
# Set it with -DCMAKE_CUDA_ARCHITECTURES=<arch> (e.g. 86 for an RTX 3060) or
# the CUDAARCHS environment variable. Left unset, CMake picks its default
# set for the CUDA toolkit in use.
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

enable_testing()
add_subdirectory(runtime)
