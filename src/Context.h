#pragma once

#include "cuda/CUDAHooksInterface.h"
#include "utils/CallOnce.h"
namespace simpletorch {
class Context {
public:
  Context() = default;
  static bool hasCUDA() {
    return c10::cuda::getCUDAHooks().hasCUDA();
  }
  static inline size_t getNumGPUs() {
    if (hasCUDA()) {
      return c10::cuda::getCUDAHooks().getNumGPUs();
    } else {
      return 0;
    }
  }
  void lazyInitCUDA() {
    c10::callOnce(cuda_init_flag_, [&]{
      c10::cuda::getCUDAHooks().initCUDA();
    });
  }
private:
  c10::OnceFlag cuda_init_flag_;
};

Context& globalContext();

static inline void init() {
  globalContext();
}
}
