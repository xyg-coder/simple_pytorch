#pragma once

#include "utils/Metaprogramming.h"
#include "utils/TypeList.h"
#include <type_traits>

namespace c10 {
class DispatchKeySet final {

};

// Given a function type, constructs a function_traits type that drops the first
// parameter type if the first parameter is of type DispatchKeySet. NB:
// DispatchKeySet is currently explicitly hidden from JIT (mainly to avoid
// pushing unnecessary arguments on the stack - see Note [ Plumbing Keys Through
// the Dispatcher] for details). If at any point in the future we need to expose
// this type to JIT, revisit the usage of this type alias.
template <class FuncType>
using remove_DispatchKeySet_arg_from_func = guts::make_function_traits_t<
  typename guts::infer_function_traits<FuncType>::return_type,
  typename std::conditional_t<
    std::is_same_v<DispatchKeySet,
      c10::guts::typelist::head_with_default<void, typename guts::infer_function_traits<FuncType>::parameter_types>>,
    guts::typelist::drop_if_nonempty_t<typename guts::infer_function_traits<FuncType>::parameter_types>, 1>,
    typename guts::infer_function_traits<FuncType>::parameter_types>>;
}
