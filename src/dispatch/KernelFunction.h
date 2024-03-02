#pragma once

#include "dispatch/DispatchKeySet.h"
#include "utils/Exception.h"
namespace c10 {

class OperatorHandle;

class KernelFunction final {
public:
  inline KernelFunction(): unboxed_kernel_func_(nullptr) {}

  explicit KernelFunction(void* unboxed_kernel_function)
    : unboxed_kernel_func_(unboxed_kernel_function) {}

  template <class Return, class... Args>
  Return call(const OperatorHandle& opHandle, DispatchKeySet dispatchKeySet, Args... args) const;

  inline bool isValidUnboxed() const {
    return unboxed_kernel_func_ != nullptr;
  }
private:
  void* unboxed_kernel_func_;
};

template <class Return, class... Args>
Return KernelFunction::call(const OperatorHandle& opHandle, DispatchKeySet dispatchKeySet, Args... args) const {
  TORCH_CHECK(isValidUnboxed(), "Trying to call an invalid unboxed kernel function");

  using FuncSignature = Return (Args...);
  FuncSignature* func = reinterpret_cast<FuncSignature*>(unboxed_kernel_func_);
  return (*func)(std::forward<Args>(args)...);
}

} // namespace c10
