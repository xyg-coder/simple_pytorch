cmake_minimum_required(VERSION 3.16)
# project(SimpleTorch LANGUAGES CUDA CXX)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 17)

# output compile_commands.json to be used by IDE
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  

file(GLOB SOURCES src/*.cpp)
file(GLOB UTILS src/utils/*.cpp)
file(GLOB CUDA_SRC src/cuda/*.cpp)

set(INCLUDES src)
find_package (glog 0.6.0 REQUIRED)

set (CMAKE_PREFIX_PATH "/usr/local/cuda" "${CMAKE_PREFIX_PATH}")  
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})  

add_executable(TestTensorExecutable ${SOURCES} example/tensor_exp.cpp)
add_executable(TestCudaContext ${SOURCES} ${CUDA_SRC} ${UTILS} example/test_cuda_context.cpp)
add_executable(TestThreadPool ${SOURCES} ${UTILS} example/test_thread_pool.cpp)
target_link_libraries (TestTensorExecutable glog::glog)
target_link_libraries (TestThreadPool glog::glog)
target_link_libraries (TestCudaContext glog::glog ${CUDA_LIBRARIES})

target_include_directories(TestTensorExecutable PRIVATE ${INCLUDES})
target_include_directories(TestThreadPool PRIVATE ${INCLUDES})
target_include_directories(TestCudaContext PRIVATE ${INCLUDES})

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
target_link_libraries (allocator_test glog::glog)

include(GoogleTest)
gtest_discover_tests(allocator_test)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61)
