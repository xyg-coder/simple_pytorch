#include "cuda/CUDAHooks.h"
#include "cuda/CUDAFunctions.h"
#include "cuda/CudaAllocator.h"
#include <glog/logging.h>

namespace c10::cuda {
int CUDAHooks::getNumGPUs() const {
  return device_count();
}

void CUDAHooks::initCUDA() const {
  device_count_ensure_non_zero();
  cuda_allocator::get()->init(getNumGPUs());
}

bool CUDAHooks::hasCUDA() const {
  return device_count() > 0;
}
}
