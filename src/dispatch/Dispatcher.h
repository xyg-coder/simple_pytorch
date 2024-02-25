#pragma once

#include "dispatch/DispatchKeySet.h"
#include "utils/TypeList.h"
namespace c10 {

template<class FuncType> class TypedOperatorHandle;
/**
 * This is a handle to an operator schema registered with the dispatcher.
 * This handle can be used to register kernels with the dispatcher or
 * to lookup a kernel for a certain set of arguments.
 */
class OperatorHandle {
public:
  OperatorHandle(OperatorHandle&&) noexcept = default;
  OperatorHandle& operator=(OperatorHandle&&) noexcept = default;
  OperatorHandle(const OperatorHandle&) = default;
  OperatorHandle& operator=(const OperatorHandle&) = default;
  // NOLINTNEXTLINE(performance-trivially-destructible)
  ~OperatorHandle();

  template<class Func>
  TypedOperatorHandle<Func> typed() const {
  }
};

/**
 * This makes sure the template receives has more than 1 types
 */
template<class FuncType>
class TypedOperatorHandle final {
  static_assert(guts::false_t<FuncType>(),
    "FuncType in OperatorHandle::typed<FuncType> was not a valid function type");
};

/**
 * This is a handle to an operator schema registered with the dispatcher.
 * It holds the same information as an OperatorHandle, but it is templated
 * on the operator arguments and allows calling the operator in an
 * unboxed way.
 */
template<class Return, class... Args>
class TypedOperatorHandle<Return (Args...)> final : public OperatorHandle { };

class Dispatcher final {
public:
  ~Dispatcher();

  template<class Return, class... Args>
  Return call(const TypedOperatorHandle<Return(Args...)>& op, Args... args) const;

  template<class Return, class... Args>
  Return redispatch(const TypedOperatorHandle<Return(Args...)>& op,
    DispatchKeySet currentDispatchKeySet, Args... args) const;
};
}
