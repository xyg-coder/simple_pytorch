cmake_minimum_required(VERSION 3.17)
# to be added back when we add .cu files
project(SimpleTorch LANGUAGES CUDA CXX)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)  

# right now for debugging
# change in the future if needed
set(CMAKE_BUILD_TYPE Debug)

# output compile_commands.json to be used by IDE
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  

file(GLOB SOURCES src/*.cpp src/utils/*.cpp src/cuda/*.cpp src/dispatch/*.cpp src/cuda_ops/*.cpp
  src/ops/*.cpp)  

set(INCLUDES src)
set (CMAKE_PREFIX_PATH "/usr/local/cuda" "${CMAKE_PREFIX_PATH}")  

find_package (glog 0.6.0 REQUIRED)
find_package(CUDAToolkit REQUIRED)  

set(ALL_LIBS glog::glog CUDA::cudart)

add_executable(TestTensorExecutable ${SOURCES} example/tensor_exp.cpp)
add_executable(TestCudaContext ${SOURCES} example/test_cuda_context.cpp)
add_executable(TestThreadPool ${SOURCES} example/test_thread_pool.cpp)
add_executable(TestException ${SOURCES} example/test_exception.cpp)
add_executable(TestAllocator ${SOURCES} example/test_allocator.cpp)
add_executable(TestDispatcher ${SOURCES} example/test_dispatcher.cpp)
add_executable(TestCudaKernel ${SOURCES} example/test_cuda_kernel.cu)
target_link_libraries (TestTensorExecutable ${ALL_LIBS})
target_link_libraries (TestThreadPool ${ALL_LIBS})
target_link_libraries (TestCudaContext ${ALL_LIBS})
target_link_libraries (TestException ${ALL_LIBS})
target_link_libraries (TestAllocator ${ALL_LIBS})
target_link_libraries (TestDispatcher ${ALL_LIBS})
target_link_libraries (TestCudaKernel ${ALL_LIBS})

target_include_directories(TestTensorExecutable PRIVATE ${INCLUDES})
target_include_directories(TestThreadPool PRIVATE ${INCLUDES})
target_include_directories(TestCudaContext PRIVATE ${INCLUDES})
target_include_directories(TestException PRIVATE ${INCLUDES})
target_include_directories (TestAllocator PRIVATE ${INCLUDES})
target_include_directories (TestDispatcher PRIVATE ${INCLUDES})
target_include_directories (TestCudaKernel PRIVATE ${INCLUDES})

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
set(TEST_LIBS ${ALL_LIBS} GTest::gtest_main)

add_executable(
  allocator_test
  ${SOURCES}
  test/allocator_test.cpp
)
add_executable(
  cuda_allocator_test
  ${SOURCES}
  test/cuda_allocator_test.cpp
)
add_executable(
  metaprogramming_test
  test/metaprogramming_test.cpp
)
add_executable(
  dispatcher_test
  ${SOURCES}
  test/dispatcher_test.cpp
)
add_executable(
  library_test
  ${SOURCES}
  test/library_test.cpp
)
target_include_directories(allocator_test PRIVATE ${INCLUDES})
target_include_directories(cuda_allocator_test PRIVATE ${INCLUDES})
target_include_directories(metaprogramming_test PRIVATE ${INCLUDES})
target_include_directories(dispatcher_test PRIVATE ${INCLUDES})
target_include_directories(library_test PRIVATE ${INCLUDES})
target_link_libraries(
  allocator_test
  ${TEST_LIBS}
)
target_link_libraries(
  cuda_allocator_test
  ${TEST_LIBS}
)
target_link_libraries(
  metaprogramming_test
  ${TEST_LIBS}
)
target_link_libraries(
  dispatcher_test
  ${TEST_LIBS}
)
target_link_libraries(
  library_test
  ${TEST_LIBS}
)

include(GoogleTest)
gtest_discover_tests(allocator_test)
gtest_discover_tests(cuda_allocator_test)
gtest_discover_tests(metaprogramming_test)
gtest_discover_tests(dispatcher_test)
gtest_discover_tests(library_test)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61)
# Set CUDA compile options, this makes us able to use static expressions in cuda
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
