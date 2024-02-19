#pragma once

#include "Device.h"
#include "utils/Exception.h"
namespace c10::cuda {
struct CudaGuard {
  explicit CudaGuard() = delete;
  explicit CudaGuard(DeviceIndex device) {
    TORCH_CHECK_WITH(NotImplementedError, device != 0,
      "CudaGuard with device != 0 is not implemented, device: ", device);
  };
};
}
