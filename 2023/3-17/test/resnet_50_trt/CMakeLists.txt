cmake_minimum_required(VERSION 2.6)

project(resnet)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

include_directories(${PROJECT_SOURCE_DIR}/include)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(/home/chaucer/Desktop/TensorRT-7.2.1.6/include/)
link_directories(/home/chaucer/Desktop/TensorRT-7.2.1.6/lib/)

add_executable(resnet50 ${PROJECT_SOURCE_DIR}/resnet50.cpp resnet50.cpp)
target_link_libraries(resnet50 nvinfer)
target_link_libraries(resnet50 cudart)

add_definitions(-O2 -pthread)

