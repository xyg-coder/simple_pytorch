#pragma once

#include "utils/Exception.h"

namespace c10::cuda {

struct CUDAHooksInterface {
  virtual ~CUDAHooksInterface() = default;
  virtual int getNumGPUs() const {
    return 0;
  }
  virtual void initCUDA() const {
    C10_THROW_ERROR(NotImplementedError, "CUDAHooks didn't implement init cuda");
  }
  virtual bool hasCUDA() const {
    return false;
  }
};

const CUDAHooksInterface& getCUDAHooks();
}
