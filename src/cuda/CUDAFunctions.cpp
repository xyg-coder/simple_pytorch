#include "cuda/CUDAFunctions.h"
#include "Device.h"
#include "cuda/CUDAException.h"
#include "macros/Macros.h"
#include "utils/Exception.h"
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
  return count;
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



// This is a codepath for CUDA 12 that comes with a critical change in behavior
// of `cudaSetDevice`. Unlike to previous CUDA versions that allocate context
// lazily CUDA 12.x eagerly allocates primary context the moment `cudaSetDevice`
// is called. This can lead to dramatic consequences and pollute the device
// memory in distributed runs. To avoid unnecessary context creation a new
// function called `MaybeSetDevice` was introduced. This function is to be
// called in device guard destructor and at the exit of torch.cuda.device
// context manager. The behavior of `MaybeSetDevice` is quite simple, it calls
// to `cudaSetDevice` if context already exist or if context was not allocated
// on targeted device it simply saves the device index. This way we can keep
// PyTorch backward compatible for applications like this:
//
// ```
// import torch
// x = torch.empty(1, device=“cuda:1”) # no CUDA context on cuda:0 after this
// call y = torch.empty(1, device=“cuda”) # CUDA context is created on cuda:0
// ```
#if CUDA_VERSION >= 12000

thread_local c10::DeviceIndex targetDeviceIndex = -1;
cudaError_t c10::cuda::GetDevice(DeviceIndex *device) {
  if (targetDeviceIndex >= 0) {
    *device = targetDeviceIndex;
    return cudaSuccess;
  }

  int tmp_device = -1;
  auto err = cudaGetDevice(&tmp_device);
  if (err == cudaSuccess) {
     TORCH_CHECK_MSG(
      tmp_device >= 0 && tmp_device <= std::numeric_limits<DeviceIndex>::max(),
      "cudaGetDevice returns invalid device ",
      tmp_device);
    *device = static_cast<DeviceIndex>(tmp_device);
    LOG_INFO("check cast, tmp_device=", tmp_device,
      ", cast result=", static_cast<DeviceIndex>(tmp_device));
  }
  return err;
}

cudaError_t c10::cuda::SetDevice(DeviceIndex device) {
  TORCH_CHECK(device >= 0, "device id must be positive");
  targetDeviceIndex = -1;
  int cur_device = -1;
  C10_CUDA_CHECK(cudaGetDevice(&cur_device));
  if (device == cur_device) {
    return cudaSuccess;
  }
  return cudaSetDevice(device);
}

void c10::cuda::SetTargetDevice() {
  if (targetDeviceIndex >= 0) {
    C10_CUDA_CHECK(SetDevice(targetDeviceIndex));
  }
}
#else
// no definition for now
#endif


c10::DeviceIndex c10::cuda::current_device() {
  DeviceIndex cur_device = -1;
  C10_CUDA_CHECK(c10::cuda::GetDevice(&cur_device));
  return cur_device;
}
