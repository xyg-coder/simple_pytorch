#pragma once

#include "Device.h"
#include <cuda_runtime_api.h>

namespace c10::cuda {
DeviceIndex device_count() noexcept;
DeviceIndex device_count_ensure_non_zero();
cudaError_t GetDeviceCount(int* dev_count);
DeviceIndex current_device();
cudaError_t GetDevice(DeviceIndex* device);
cudaError_t SetDevice(DeviceIndex device);
void SetTargetDevice();
}
