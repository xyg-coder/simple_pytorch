#include "cuda_ops/EmptyTensor.h"
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
};

TORCH_LIBRARY_IMPL(aten, CUDA, m) {
  LOG_INFO("Registering impls to dispatcher, dispatchKey=CUDA");
  m.impl(
    FunctionSchema(FunctionSchema::EMPTY, OperatorName("empty", "")),
    simpletorch::impl::empty_cuda);
};
}
