# /CMakeList.txt : Top-level CMake file

cmake_minimum_required (VERSION 3.18)

project(HammingOneProject LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

# Add shared library
add_subdirectory("shared")

# Include sub-projects.
add_subdirectory ("HammingOne")
add_subdirectory ("HammingOne_Naive")
