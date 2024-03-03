#include "Allocator.h"
#include "Context.h"
#include "dispatch/DispatchKey.h"
#include "dispatch/DispatchKeySet.h"
#include "dispatch/Dispatcher.h"
#include "dispatch/FunctionSchema.h"
#include "dispatch/KernelFunction.h"
#include "dispatch/OperatorName.h"
#include "dispatch/RegistrationHandleRAII.h"
#include "cuda/CudaAllocator.h"
#include <cstddef>
#include <iostream>
#include <utility>
#include <glog/logging.h>


int test_sum(int a, int b) {
  return a + b;
}

int test_sum_higher(int a, int b) {
  return a + b + 1;
}

int test_sum_highest(int a, int b) {
  return a + b + 2;
}

c10::DataPtr allocate_memory(size_t size) {
  simpletorch::globalContext().lazyInitCUDA();
  c10::cuda::cuda_allocator::CUDAAllocator *allocator = c10::cuda::cuda_allocator::get();
  c10::DataPtr ptr = allocator->allocate(size);
  return ptr;
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
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
    c10::KernelFunction((void*)test_sum),
    std::nullopt,
    "test-debug-register-impl");

  c10::RegistrationHandleRAII raii_autograd = c10::Dispatcher::singleton().registerImpl(
    c10::OperatorName("TEST", ""),
    c10::DispatchKey::Autograd,
    c10::KernelFunction((void*)test_sum_higher),
    std::nullopt,
    "test-debug-register-impl");

    c10::RegistrationHandleRAII cpu = c10::Dispatcher::singleton().registerImpl(
    c10::OperatorName("TEST", ""),
    c10::DispatchKey::CPU,
    c10::KernelFunction((void*)test_sum_highest),
    std::nullopt,
    "test-debug-register-impl");

  c10::TypedOperatorHandle handle = c10::Dispatcher::singleton().findOp(c10::OperatorName("TEST", ""))->typed<int(int, int)>();
  int a = handle.call(10, 20);
  std::cout << a << std::endl;

  c10::RegistrationHandleRAII allocator_schema_register = c10::Dispatcher::singleton().registerDef(
    c10::FunctionSchema(
        c10::FunctionSchema::ALLOCATOR,
        c10::OperatorName("TEST_ALLOCATOR", "")), "test-debug-allocator`");
  
  c10::RegistrationHandleRAII allocator_cuda = c10::Dispatcher::singleton().registerImpl(
    c10::OperatorName("TEST_ALLOCATOR", ""),
    c10::DispatchKey::CUDA,
    c10::KernelFunction((void*)allocate_memory),
    std::nullopt,
    "test-debug-register-impl");

  c10::DispatchKeySet keyset;
  keyset = keyset.add(c10::DispatchKey::CUDA);

  a = handle.redispatch(keyset, 10, 20);
  std::cout << a << std::endl;

  keyset = keyset.add(c10::DispatchKey::CPU);
  a = handle.redispatch(keyset, 10, 20);
  std::cout << a << std::endl;

    keyset = keyset.add(c10::DispatchKey::Autograd);
  a = handle.redispatch(keyset, 10, 20);
  std::cout << a << std::endl;

  keyset = c10::DispatchKeySet(c10::DispatchKeySet::FULL);
  a = handle.redispatch(keyset, 10, 20);
  std::cout << a << std::endl;
  
  c10::TypedOperatorHandle allocator_handle = c10::Dispatcher::singleton().findOp(c10::OperatorName("TEST_ALLOCATOR", ""))->
  typed<c10::DataPtr(size_t)>();
  c10::DataPtr ptr = allocator_handle.call(64 * 64 * sizeof(int));
}
