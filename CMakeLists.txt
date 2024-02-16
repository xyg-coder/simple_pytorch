cmake_minimum_required(VERSION 3.17)
# to be added back when we add .cu files
project(SimpleTorch LANGUAGES CUDA CXX)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)  

# output compile_commands.json to be used by IDE
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  

file(GLOB SOURCES src/*.cpp src/utils/*.cpp src/cuda/*.cpp)  

set(INCLUDES src)
set (CMAKE_PREFIX_PATH "/usr/local/cuda" "${CMAKE_PREFIX_PATH}")  

find_package (glog 0.6.0 REQUIRED)
find_package(CUDAToolkit REQUIRED)  

set(ALL_LIBS glog::glog CUDA::cudart)

add_executable(TestTensorExecutable ${SOURCES} example/tensor_exp.cpp)
add_executable(TestCudaContext ${SOURCES} example/test_cuda_context.cpp)
add_executable(TestThreadPool ${SOURCES} example/test_thread_pool.cpp)
add_executable(TestException ${SOURCES} example/test_exception.cpp)
target_link_libraries (TestTensorExecutable ${ALL_LIBS})
target_link_libraries (TestThreadPool ${ALL_LIBS})
target_link_libraries (TestCudaContext ${ALL_LIBS})
target_link_libraries (TestException ${ALL_LIBS})

target_include_directories(TestTensorExecutable PRIVATE ${INCLUDES})
target_include_directories(TestThreadPool PRIVATE ${INCLUDES})
target_include_directories(TestCudaContext PRIVATE ${INCLUDES})
target_include_directories(TestException PRIVATE ${INCLUDES})

# TEST
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  allocator_test
  ${SOURCES}
  test/allocator_test.cpp
)
target_include_directories(allocator_test PRIVATE ${INCLUDES})
target_link_libraries(
  allocator_test
  GTest::gtest_main
)
target_link_libraries (allocator_test ${ALL_LIBS})

include(GoogleTest)
gtest_discover_tests(allocator_test)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61)
