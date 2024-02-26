#pragma once

#include "dispatch/DispatchKeySet.h"
#include "dispatch/OperatorEntry.h"
#include "dispatch/OperatorName.h"
#include "utils/TypeList.h"
#include <optional>
#include <unordered_map>
namespace c10 {

template<class FuncType> class TypedOperatorHandle;

struct OperatorDef final {
  explicit OperatorDef(OperatorName&& operator_name)
    :op_(std::move(operator_name)) {};

  OperatorEntry op_;

    // These refer to the number of outstanding RegistrationHandleRAII
    // for this operator.  def_count reflects only def() registrations
    // (in the new world, this should only ever be 1, but old style
    // registrations may register the schema multiple times, which
    // will increase this count).  def_and_impl_count reflects the number
    // of combined def() and impl() registrations.  When the last def() gets
    // unregistered, we must immediately call the Deregistered listeners, but we
    // must not actually delete the handle as there are other outstanding RAII
    // destructors which will try to destruct and they had better still have a
    // working operator handle in this case
    size_t def_count = 0;
    size_t def_and_impl_count = 0;
};

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

  bool hasSchema() const;

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
class TypedOperatorHandle<Return (Args...)> final : public OperatorHandle {
public:
  TypedOperatorHandle(TypedOperatorHandle&&) noexcept = default;
  TypedOperatorHandle& operator=(TypedOperatorHandle&&) noexcept = default;
  TypedOperatorHandle(const TypedOperatorHandle&) = default;
  TypedOperatorHandle& operator=(const TypedOperatorHandle&) = default;

  Return call(Args... args) const {}

  Return redispatch(DispatchKeySet currentDispatchKeySet, Args... args) const {}

};

class Dispatcher final {
public:
  ~Dispatcher();

  template<class Return, class... Args>
  Return call(const TypedOperatorHandle<Return(Args...)>& op, Args... args) const;

  template<class Return, class... Args>
  Return redispatch(const TypedOperatorHandle<Return(Args...)>& op,
    DispatchKeySet currentDispatchKeySet, Args... args) const;

  std::optional<OperatorHandle> findSchema(const OperatorName& operator_name);

  OperatorHandle findSchemaOrThrow(const char* name, const char* overload_name);

  std::optional<OperatorHandle> findOp(const OperatorName& operator_name);

private:
  std::unordered_map<typename Key, typename Tp>
};
}
