#include "cuda/CUDAHooks.h"
#include "cuda/CUDAFunctions.h"
#include <glog/logging.h>

namespace c10::cuda {
int CUDAHooks::getNumGPUs() const {
  return c10::cuda::device_count();
}

void CUDAHooks::initCUDA() const {
  LOG(ERROR) << "we should call c10::cuda::CUDACachingAllocator::init(num_devices); in the init";
  c10::cuda::device_count_ensure_non_zero();
}

bool CUDAHooks::hasCUDA() const {
  return c10::cuda::device_count() > 0;
}
}
