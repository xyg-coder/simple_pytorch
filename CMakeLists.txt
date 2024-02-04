cmake_minimum_required(VERSION 3.12)
project(TestTensor)

set(CMAKE_CXX_STANDARD 17)

# output compile_commands.json to be used by IDE
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  

file(GLOB SOURCES src/*.cpp)

set(INCLUDES src)

add_executable(TestTensorExecutable ${SOURCES} example/tensor_exp.cpp)

target_include_directories(TestTensorExecutable PRIVATE ${INCLUDES})

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
  tensor_test
  ${SOURCES}
  test/tensor_test.cpp
)
target_include_directories(tensor_test PRIVATE ${INCLUDES})
target_link_libraries(
  tensor_test
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(tensor_test)
