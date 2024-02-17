#include "cuda/CUDAFunctions.h"
#include "Device.h"
#include "cuda/CUDAException.h"
#include "macros/Macros.h"
#include "utils/Logging.h"
#include <exception>
#include <glog/logging.h>
#include <stdexcept>

cudaError_t c10::cuda::GetDeviceCount(int* dev_count) {
  return cudaGetDeviceCount(dev_count);
}

int device_count_impl(bool fail_if_no_driver) {
  int count = 0;
  auto err = C10_CUDA_ERROR_HANDLED(c10::cuda::GetDeviceCount(&count));
  if (err == cudaSuccess) {
    return count;
  }
  // Clear out the error state, so we don't spuriously trigger someone else.
  // (This shouldn't really matter, since we won't be running very much CUDA
  // code in this regime.)
  cudaError_t last_err C10_UNUSED = cudaGetLastError();
  switch (err) {
    case cudaErrorNoDevice:
      count = 0;
      break;
    default:
      throw std::logic_error(cudaGetErrorString(err));
  }
}

c10::DeviceIndex c10::cuda::device_count_ensure_non_zero() {
  int count = device_count_impl(true);
  if(count == 0) {
    throw std::logic_error("NO CUDA GPUs are available");
  }

  return static_cast<DeviceIndex>(count);
}

c10::DeviceIndex c10::cuda::device_count() noexcept {
  // static initialization
  static int count = [](){
    try {
      auto result = device_count_impl(false);
      return result;
    } catch(std::exception& e) {
      LOG_ERROR("count device error, We silenced the error here");
      return 0;
    }
  }();
  return static_cast<DeviceIndex>(count);
}
