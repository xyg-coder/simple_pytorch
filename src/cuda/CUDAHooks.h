#pragma once

#include "Device.h"
#include "cuda/CUDAHooksInterface.h"

namespace c10::cuda {
struct CUDAHooks : public CUDAHooksInterface {
  int getNumGPUs() const override;
  void initCUDA() const override;
  bool hasCUDA() const override;
};
}
