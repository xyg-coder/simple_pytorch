#pragma once

#include "Device.h"
#include <stdexcept>

namespace c10::cuda {

struct CUDAHooksInterface {
  virtual ~CUDAHooksInterface() = default;
  virtual int getNumGPUs() const {
    return 0;
  }
  virtual void initCUDA() const {
    throw std::logic_error("CUDAHooks didn't implement init cuda");
  }
  virtual bool hasCUDA() const {
    return false;
  }
};

const CUDAHooksInterface& getCUDAHooks();
}
