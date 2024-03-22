#include "cuda/CUDAException.h"
#include "macros/Macros.h"
#include "utils/Exception.h"
#include <cuda_runtime.h>
#include <string>

namespace c10::cuda {
void c10_cuda_check_implementation(
  const int32_t err,
  const char* filename,
  const char* function_name,
  const int line_number,
  const bool include_device_assertions) {

  const auto cuda_error = static_cast<cudaError_t>(err);
  TORCH_CHECK_WITH(NotImplementedError, !include_device_assertions, "Device assertion is not implemented yet");
  const auto cuda_kernel_failure = false;
  if (C10_LIKELY(cuda_error == cudaSuccess && !cuda_kernel_failure)) {
    return;
  }

  // clean last error
  cudaGetLastError();

  std::string check_message;
  check_message.append("CUDA error: ");
  check_message.append(cudaGetErrorString(cuda_error));
  check_message.append("\n");
  // so we can pass down the filenames
  torchCheckFail(
    function_name,
    filename,
    line_number,
    TORCH_CHECK_MSG(false, "", check_message)
  );
}
}
