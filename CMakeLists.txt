cmake_minimum_required(VERSION 3.12)
project(TestTensor)

set(CMAKE_CXX_STANDARD 17)

file(GLOB SOURCES src/*.cpp)

set(INCLUDES src)

add_executable(TestTensorExecutable ${SOURCES} example/tensor_exp.cpp)

target_include_directories(TestTensorExecutable PRIVATE ${INCLUDES})
