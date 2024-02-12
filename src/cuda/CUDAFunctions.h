#pragma once

#include "Device.h"
#include <cuda_runtime_api.h>

namespace c10::cuda {
DeviceIndex device_count() noexcept;
DeviceIndex device_count_ensure_non_zero();
cudaError_t GetDeviceCount(int* dev_count);
}
