#pragma once

#include "Device.h"
namespace c10::cuda {
struct CudaGuard {
  explicit CudaGuard() = delete;
  explicit CudaGuard(DeviceIndex) {};
};
}
