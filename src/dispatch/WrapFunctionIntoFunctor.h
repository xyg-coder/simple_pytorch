#pragma once

#include "dispatch/OperatorKernel.h"
#include "utils/Metaprogramming.h"
#include "utils/TypeList.h"
#include <utility>

namespace c10 {

template<class FuncType, class ReturnType, class ParameterList> class WrapFunctionIntoRuntimeFunctor_ {};

template<class FuncType, class ReturnType, class... Parameters>
class WrapFunctionIntoRuntimeFunctor_<FuncType, ReturnType, guts::typelist::typelist<Parameters...>> final 
  : public OperatorKernel {
public:
  template<class FuncType_>
  explicit WrapFunctionIntoRuntimeFunctor_(FuncType_&& kernel_func)
    : kernel_func_(std::forward<FuncType_>(kernel_func)) { }

  ReturnType operator()(Parameters... args) {
    return kernel_func_(std::forward<Parameters>(args)...);
  }
private:
  FuncType kernel_func_;
};

template<class FuncType>
using WrapFunctionIntoRuntimeFunctor = WrapFunctionIntoRuntimeFunctor_<
  FuncType,
  typename guts::infer_function_traits_t<FuncType>::return_type,
  typename guts::infer_function_traits_t<FuncType>::parameter_types>;
}
