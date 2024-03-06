#include "Allocator.h"
#include "Context.h"
#include "cuda/CudaAllocator.h"
#include "dispatch/DispatchKey.h"
#include "dispatch/DispatchKeySet.h"
#include "dispatch/Dispatcher.h"
#include "dispatch/RegistrationHandleRAII.h"
#include <gtest/gtest.h>

c10::DataPtr allocate_memory(size_t size) {
  simpletorch::globalContext().lazyInitCUDA();
  c10::cuda::cuda_allocator::CUDAAllocator *allocator = c10::cuda::cuda_allocator::get();
  c10::DataPtr ptr = allocator->allocate(size);
  return ptr;
}

TEST(DispatcherTest, TestAllocator) {
  c10::RegistrationHandleRAII allocator_schema_register = c10::Dispatcher::singleton().registerDef(
    c10::FunctionSchema(
        c10::FunctionSchema::ALLOCATOR,
        c10::OperatorName("TEST_ALLOCATOR", "")), "test-debug-allocator`");
  
  c10::RegistrationHandleRAII allocator_cuda = c10::Dispatcher::singleton().registerImpl(
    c10::OperatorName("TEST_ALLOCATOR", ""),
    c10::DispatchKey::CUDA,
    c10::KernelFunction::makeFromUnboxedRuntimeFunction(allocate_memory),
    std::nullopt,
    "test-debug-register-impl");
  
  c10::TypedOperatorHandle allocator_handle = c10::Dispatcher::singleton().findOp(c10::OperatorName("TEST_ALLOCATOR", ""))->
    typed<c10::DataPtr(size_t)>();
  c10::DataPtr ptr = allocator_handle.call(64 * 64 * sizeof(int));
}

int test_sum(int a, int b) {
  return a + b;
}

int test_sum_higher(int a, int b) {
  return a + b + 1;
}

int test_sum_highest(int a, int b) {
  return a + b + 2;
}

TEST(DispatcherTest, callind_correct_function) {
    c10::RegistrationHandleRAII raii = c10::Dispatcher::singleton()
    .registerDef(
      c10::FunctionSchema(
        c10::FunctionSchema::TEST,
        c10::OperatorName("TEST", "")
      ),
      "test-debug"
    );

  c10::RegistrationHandleRAII raii_cuda = c10::Dispatcher::singleton().registerImpl(
    c10::OperatorName("TEST", ""),
    c10::DispatchKey::CUDA,
    c10::KernelFunction::makeFromUnboxedRuntimeFunction(test_sum_higher),
    std::nullopt,
    "test-debug-register-impl");

  c10::RegistrationHandleRAII raii_autograd = c10::Dispatcher::singleton().registerImpl(
    c10::OperatorName("TEST", ""),
    c10::DispatchKey::Autograd,
    c10::KernelFunction::makeFromUnboxedRuntimeFunction(test_sum_highest),
    std::nullopt,
    "test-debug-register-impl");

    c10::RegistrationHandleRAII cpu = c10::Dispatcher::singleton().registerImpl(
    c10::OperatorName("TEST", ""),
    c10::DispatchKey::CPU,
    c10::KernelFunction::makeFromUnboxedRuntimeFunction(test_sum),
    std::nullopt,
    "test-debug-register-impl");

  c10::TypedOperatorHandle handle = c10::Dispatcher::singleton().findOp(c10::OperatorName("TEST", ""))->typed<int(int, int)>();
  int a = handle.call(10, 20);
  // call auto-grad
  EXPECT_EQ(a, 32);

  c10::RegistrationHandleRAII allocator_schema_register = c10::Dispatcher::singleton().registerDef(
    c10::FunctionSchema(
        c10::FunctionSchema::ALLOCATOR,
        c10::OperatorName("TEST_ALLOCATOR", "")), "test-debug-allocator`");
  
  c10::RegistrationHandleRAII allocator_cuda = c10::Dispatcher::singleton().registerImpl(
    c10::OperatorName("TEST_ALLOCATOR", ""),
    c10::DispatchKey::CUDA,
    c10::KernelFunction::makeFromUnboxedRuntimeFunction(allocate_memory),
    std::nullopt,
    "test-debug-register-impl");

  c10::DispatchKeySet keyset;
  keyset = keyset.add(c10::DispatchKey::CUDA);

  a = handle.redispatch(keyset, 10, 20);
  // call gpu
  EXPECT_EQ(a, 31);

  keyset = keyset.add(c10::DispatchKey::CPU);
  a = handle.redispatch(keyset, 10, 20);
  // still call gpu
  EXPECT_EQ(a, 31);

  keyset = keyset.add(c10::DispatchKey::Autograd);
  a = handle.redispatch(keyset, 10, 20);
  // call autograd
  EXPECT_EQ(a, 32);

  keyset = c10::DispatchKeySet(c10::DispatchKey::CPU);
  a = handle.redispatch(keyset, 10, 20);
  // call cpu
  EXPECT_EQ(a, 30);

  keyset = c10::DispatchKeySet(c10::DispatchKeySet::FULL);
  a = handle.redispatch(keyset, 10, 20);
  // call cpu
  EXPECT_EQ(a, 32);
}
