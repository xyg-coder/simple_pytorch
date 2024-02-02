cmake_minimum_required(VERSION 3.12)
project(TestTensor)

set(CMAKE_CXX_STANDARD 17)

# output compile_commands.json to be used by IDE
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  

file(GLOB SOURCES src/*.cpp)

set(INCLUDES src)

add_executable(TestTensorExecutable ${SOURCES} example/tensor_exp.cpp)

target_include_directories(TestTensorExecutable PRIVATE ${INCLUDES})
