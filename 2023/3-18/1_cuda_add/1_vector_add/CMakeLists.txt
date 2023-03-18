cmake_minimum_required(VERSION 2.6)
project(vector_add)

set(CMAKE_CXX_STANDARD 11)
find_package(CUDA REQUIRED)

cuda_add_executable(vector_add vector_add.cu)
target_link_libraries(vector_add)
