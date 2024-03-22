#pragma once

#include "utils/Exception.h"
#include <cstdint>
#include <cuda.h>

namespace c10::cuda {

class CUDAError : public c10::Error {
  using Error::Error;
};

void c10_cuda_check_implementation(
    const int32_t err,
    const char* filename,
    const char* function_name,
    const int line_number,
    const bool include_device_assertions);
}

#define C10_CUDA_ERROR_HANDLED(EXPR) EXPR

// TODO: once we implement the device assertion, change to true
#define C10_CUDA_CHECK(EXPR)                  \
  do {                                        \
    const cudaError_t __err = EXPR;           \
    c10::cuda::c10_cuda_check_implementation( \
    static_cast<int32_t>(__err),              \
    __FILE__,                                 \
    __func__,                                 \
    static_cast<uint32_t>(__LINE__),          \
    false);                                   \
  } while(0)

// This should be used directly after every kernel launch to ensure
// the launch happened correctly and provide an early, close-to-source
// diagnostic if it didn't.
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
