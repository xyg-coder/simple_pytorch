#pragma once

#include "dispatch/DispatchKey.h"
#include "utils/Metaprogramming.h"
#include "utils/TypeList.h"
#include <cstdint>
#include <type_traits>

namespace c10 {
class DispatchKeySet final {
public:
  enum Full { FULL };
  enum Raw { RAW };
  constexpr DispatchKeySet() = default;
  // TODO: revisit this, seems right now the least importantce is using 000
  constexpr DispatchKeySet(Full)
    : repr_((1ULL << (num_functionality_keys - 1)) - 1) {}
  constexpr DispatchKeySet(Raw, uint64_t repr) : repr_(repr) {}
  constexpr DispatchKeySet(uint64_t repr) : repr_(repr) {}
  constexpr DispatchKeySet operator&(DispatchKeySet other) const {
    return DispatchKeySet(repr_ & other.repr_);
  }

  // returns the DispatchKey of highest priority in the set.
  DispatchKey highestPriorityTypeId() const;
  // returns the index in the operator table of highest priority key in the the keyset
  int getDispatchTableIndexForDispatchKeySet() const;
private:
  uint64_t repr_ = 0;
};



// Given a function type, constructs a function_traits type that drops the first
// parameter type if the first parameter is of type DispatchKeySet. NB:
// DispatchKeySet is currently explicitly hidden from JIT (mainly to avoid
// pushing unnecessary arguments on the stack - see Note [ Plumbing Keys Through
// the Dispatcher] for details). If at any point in the future we need to expose
// this type to JIT, revisit the usage of this type alias.
template <class FuncType>
using remove_DispatchKeySet_arg_from_func = guts::make_function_traits_t<
  typename guts::infer_function_traits_t<FuncType>::return_type,
  typename std::conditional_t<
    std::is_same_v<DispatchKeySet,
      c10::guts::typelist::head_with_default_t<void, typename guts::infer_function_traits_t<FuncType>::parameter_types>>,
    c10::guts::typelist::drop_if_nonempty_t<typename guts::infer_function_traits_t<FuncType>::parameter_types, 1>,
    typename guts::infer_function_traits_t<FuncType>::parameter_types>>;
}
