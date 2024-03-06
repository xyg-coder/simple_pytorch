#pragma once

#include "dispatch/DispatchKeySet.h"
#include "dispatch/OperatorKernel.h"
#include "utils/Metaprogramming.h"
#include "utils/TypeList.h"
#include <type_traits>
#include <utility>
namespace c10 {

template<class KernelFunctor, class OpSignature>
struct wrap_kernel_functor_unboxed_ final {};

template <class KernelFunctor, class ReturnType, class... ParameterTypes>
struct wrap_kernel_functor_unboxed_<KernelFunctor, ReturnType(ParameterTypes...)> final {
  static_assert(
    std::is_same<ReturnType,
      typename guts::infer_function_traits_t<KernelFunctor>::return_type>::value,
    "ReturnType mismatch.");
  static_assert(std::is_same<guts::typelist::typelist<ParameterTypes...>,
      typename guts::infer_function_traits_t<KernelFunctor>::parameter_types>::value,
    "Parameter types mismatch");
  
  static ReturnType call(OperatorKernel* functor, DispatchKeySet, ParameterTypes... args) {
    KernelFunctor* functor_ = static_cast<KernelFunctor*>(functor);
    return (*functor_)(std::forward<ParameterTypes>(args)...);
  }
};

// this specialization is for kernels with a first argument of type DispatchKeySet
template<class KernelFunctor, class ReturnType, class... ParameterTypes>
struct wrap_kernel_functor_unboxed_<KernelFunctor, ReturnType(DispatchKeySet, ParameterTypes...)> final {
  static_assert(std::is_same<ReturnType,
      typename guts::infer_function_traits_t<KernelFunctor>::return_type>::value,
    "Return type mismatch");
  static_assert(std::is_same<guts::typelist::typelist<DispatchKeySet, ParameterTypes...>,
      typename guts::infer_function_traits_t<KernelFunctor>::parameter_types>::value,
    "Parameter types mismatch");

  // See [Note: Argument forwarding in the dispatcher] for why ParameterTypes doesn't use &&
  static ReturnType call(OperatorKernel* functor, DispatchKeySet dispatchKeySet, ParameterTypes... args) {
    KernelFunctor* functor_ = static_cast<KernelFunctor*>(functor);
    // We're explicitly taking in a dispatchKeySet and forwarding it to the registered kernel.
    // See Note [Plumbing Keys Through The Dispatcher 2] for details.
    return (*functor_)(dispatchKeySet, std::forward<ParameterTypes>(args)...);
  }
};

  template<class KernelFunctor>
  using wrap_kernel_functor_unboxed = wrap_kernel_functor_unboxed_<KernelFunctor, typename guts::infer_function_traits_t<KernelFunctor>::func_type>;
}
