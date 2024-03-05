#pragma once

#include "Device.h"
#include "utils/Exception.h"
namespace c10::cuda {
struct CudaGuard {
  explicit CudaGuard() = delete;
  explicit CudaGuard(DeviceIndex device) {
    // 0 means the first gpu, -1 means not initialized yet
    TORCH_CHECK_WITH(NotImplementedError, device == 0 || device == -1,
      "CudaGuard with deviceIndex != 0 is not implemented, device: ", device);
  };
  explicit CudaGuard(Device device): CudaGuard(device.index()) {
    TORCH_CHECK_WITH(NotImplementedError, device.is_cuda(),
      "Currently only cuda device with index=0 is supported. device=", device);
  }
};
}
