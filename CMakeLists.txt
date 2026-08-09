cmake_minimum_required(VERSION 3.24)

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  # Older CMake versions fail to auto-detect newer CUDA toolkits.
  if(EXISTS /usr/local/cuda/bin/nvcc)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
  else()
    message(FATAL_ERROR "WarpVM requires a CUDA compiler (nvcc)")
  endif()
endif()

# Must precede project(): CUDA compiler detection caches the architecture
# list during enable_language(CUDA) and defaults badly on unknown toolkits.
# RTX 3060 = sm_86. Override with -DCMAKE_CUDA_ARCHITECTURES=<cc>.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(WarpVM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

enable_testing()
add_subdirectory(runtime)
