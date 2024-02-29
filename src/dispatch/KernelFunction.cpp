#include "dispatch/KernelFunction.h"
#include "utils/Exception.h"
#include <utility>

namespace c10 {

template <class Return, class... Args>
Return KernelFunction::call(const OperatorHandle& opHandle, DispatchKeySet dispatchKeySet, Args... args) {
  TORCH_CHECK(isValidUnboxed(), "Trying to call an invalid unboxed kernel function");

  using FuncSignature = Return (Args...);
  FuncSignature* func = reinterpret_cast<FuncSignature*>(unboxed_kernel_func_);
  return (*func)(std::forward<Args>(args)...);
}

}
