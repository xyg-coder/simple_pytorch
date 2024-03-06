#pragma once

#include "dispatch/DispatchKeySet.h"
#include "dispatch/MakeBoxedFromUnboxedFunctor.h"
#include "dispatch/OperatorKernel.h"
#include "dispatch/WrapFunctionIntoFunctor.h"
#include "utils/Exception.h"
#include "utils/TypeTraits.h"
#include <memory>
#include <type_traits>
#include <utility>

namespace c10 {

class OperatorHandle;

class KernelFunction final {
public:
  inline KernelFunction(): unboxed_kernel_func_(nullptr) {}

  explicit KernelFunction(std::shared_ptr<OperatorKernel> functor, void* unboxed_kernel_function)
    : unboxed_kernel_func_(unboxed_kernel_function), functor_(std::move(functor)) {
      auto debugKernel = functor.get();
    }

  template <class Return, class... Args>
  Return call(const OperatorHandle& opHandle, DispatchKeySet dispatchKeySet, Args... args) const;

  inline bool isValidUnboxed() const {
    return unboxed_kernel_func_ != nullptr;
  }

  template<class FuncType>
  static KernelFunction makeFromUnboxedRuntimeFunction(FuncType* func) {
    static_assert(guts::is_function_type<FuncType>::value,
      "Tried to call KernelFunction::makeFromUnboxedRuntimeFunction with a non-function type.");

    using FunctorType = WrapFunctionIntoRuntimeFunctor<std::decay_t<FuncType>> ;
    std::shared_ptr<OperatorKernel> kernel(new FunctorType(std::forward<FuncType*>(func)));

    return makeFromUnboxedFunctor<FunctorType>(std::move(kernel));
  }

  template<class KernelFunctor>
  static KernelFunction makeFromUnboxedFunctor(std::shared_ptr<OperatorKernel> kernelFunctor) {
    static_assert(guts::is_functor<KernelFunctor>::value,
      "Tried to call KernelFunction::makeFromUnboxedFunctor<KernelFunctor> but the argument is not a functor.");
    static_assert(std::is_base_of_v<OperatorKernel, KernelFunctor>,
      "Tried to call KernelFunction::makeFromUnboxedFunctor<KernelFunctor>, but the functor doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");
    
    auto* unboxed_functor = &wrap_kernel_functor_unboxed<KernelFunctor>::call;
    void* void_unboxed_fn = reinterpret_cast<void*>(unboxed_functor);  
    return KernelFunction(std::move(kernelFunctor), void_unboxed_fn);
  }

private:
  // this stores one pointer to wrap_kernel_functor_unboxed
  void* unboxed_kernel_func_;
  std::shared_ptr<OperatorKernel> functor_;
};

template <class Return, class... Args>
Return KernelFunction::call(const OperatorHandle& opHandle, DispatchKeySet dispatchKeySet, Args... args) const {
  TORCH_CHECK(isValidUnboxed(), "Trying to call an invalid unboxed kernel function");

  using FuncSignature = Return (OperatorKernel*, DispatchKeySet, Args...);
  FuncSignature* func = reinterpret_cast<FuncSignature*>(unboxed_kernel_func_);
  return (*func)(functor_.get(), dispatchKeySet, std::forward<Args>(args)...);
}

} // namespace c10
