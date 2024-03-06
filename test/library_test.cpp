#include "Allocator.h"
#include "Context.h"
#include "cuda/CudaAllocator.h"
#include "dispatch/Dispatcher.h"
#include "dispatch/FunctionSchema.h"
#include "dispatch/Library.h"
#include <gtest/gtest.h>

namespace c10 {
namespace {

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

TORCH_LIBRARY(aten, m) {
  m.def(FunctionSchema(FunctionSchema::TEST,
    OperatorName("test", "library_test")));
  m.def(FunctionSchema(FunctionSchema::ALLOCATOR,
    OperatorName("allocator", "library_test")));
};

TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl(
    FunctionSchema(FunctionSchema::TEST, OperatorName("test", "library_test")),
    test_sum);
};

TORCH_LIBRARY_IMPL(aten, CUDA, m) {
  m.impl(
    FunctionSchema(FunctionSchema::TEST, OperatorName("test", "library_test")),
    test_sum_higher);
  m.impl(FunctionSchema(FunctionSchema::ALLOCATOR,
    OperatorName("allocator", "library_test")), allocate_memory);
};

TORCH_LIBRARY_IMPL(aten, Autograd, m) {
  m.impl(
    FunctionSchema(FunctionSchema::TEST, OperatorName("test", "library_test")),
    test_sum_highest);
};
}

TEST(LibraryTest, TestAllocator) {
    c10::TypedOperatorHandle allocator_handle = c10::Dispatcher::singleton().findOp(c10::OperatorName("allocator", "library_test"))->
    typed<c10::DataPtr(size_t)>();
  c10::DataPtr ptr = allocator_handle.call(64 * 64 * sizeof(int));
}

}
