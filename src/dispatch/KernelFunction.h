#pragma once

#include "dispatch/DispatchKeySet.h"
#include "dispatch/Dispatcher.h"
namespace c10 {
class KernelFunction final {
public:
  inline KernelFunction(): unboxed_kernel_func_(nullptr) {}

  explicit KernelFunction(void* unboxed_kernel_function)
    : unboxed_kernel_func_(unboxed_kernel_function) {}

  template <class Return, class... Args>
  Return call(const OperatorHandle& opHandle, DispatchKeySet dispatchKeySet, Args... args);

  inline bool isValidUnboxed() const {
    return unboxed_kernel_func_ != nullptr;
  }
private:
  void* unboxed_kernel_func_;
};
}
