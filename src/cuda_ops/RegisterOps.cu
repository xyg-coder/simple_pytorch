#include "cuda_ops/EmptyTensor.h"
#include "cuda_ops/FillKernel.cuh"
#include "cuda_ops/TensorFactories.h"
#include "dispatch/DispatchKey.h"
#include "dispatch/FunctionSchema.h"
#include "dispatch/Library.h"
#include "dispatch/OperatorName.h"
#include "utils/Logging.h"

namespace c10 {

TORCH_LIBRARY(aten, m) {
  LOG_INFO("Registering defs to dispatcher");
  m.def(FunctionSchema(FunctionSchema::EMPTY,
    OperatorName("empty", "")));
  m.def(FunctionSchema(FunctionSchema::FILL, OperatorName("fill", "")));
  m.def(FunctionSchema(FunctionSchema::FILL_TENSOR, OperatorName("tensor_fill", "")));
};

TORCH_LIBRARY_IMPL(aten, CUDA, m) {
  LOG_INFO("Registering impls to dispatcher, dispatchKey=CUDA");
  m.impl(
    FunctionSchema(FunctionSchema::EMPTY, OperatorName("empty", "")),
    simpletorch::impl::empty_cuda);
  m.impl(
    FunctionSchema(FunctionSchema::FILL, OperatorName("fill", "")),
    simpletorch::impl::fill_out);
  m.impl(
    FunctionSchema(FunctionSchema::FILL_TENSOR, OperatorName("tensor_fill", "")),
    simpletorch::impl::tensor_fill);
};
}
