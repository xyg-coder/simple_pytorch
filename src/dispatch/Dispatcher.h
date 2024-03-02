#pragma once

#include "dispatch/CppSignature.h"
#include "dispatch/DispatchKey.h"
#include "dispatch/DispatchKeySet.h"
#include "dispatch/FunctionSchema.h"
#include "dispatch/KernelFunction.h"
#include "dispatch/OperatorEntry.h"
#include "dispatch/OperatorName.h"
#include "dispatch/RegistrationHandleRAII.h"
#include "utils/LeftRight.h"
#include "utils/TypeList.h"
#include <atomic>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
namespace c10 {

template<class FuncType> class TypedOperatorHandle;
class Dispatcher;

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
  friend class Dispatcher;
  OperatorHandle(OperatorHandle&&) noexcept = default;
  OperatorHandle& operator=(OperatorHandle&&) noexcept = default;
  OperatorHandle(const OperatorHandle&) = default;
  OperatorHandle& operator=(const OperatorHandle&) = default;
  // NOLINTNEXTLINE(performance-trivially-destructible)
  ~OperatorHandle() = default;

  template<class Func>
  TypedOperatorHandle<Func> typed() const {
    operator_def_->op_.assertSignatureIsCorrect<Func>();
    return TypedOperatorHandle<Func>(operators_iterator_);
  }

  bool hasSchema() const {
    return operator_def_->op_.hasSchema();
  }

  const FunctionSchema& schema() const {
    return operator_def_->op_.schema();
  }

  const OperatorName& operator_name() const {
    return operator_def_->op_.operatorName();
  }

protected:
  explicit OperatorHandle(std::list<OperatorDef>::iterator operator_iterator)
    :operator_def_(&*operator_iterator),
      operators_iterator_(operator_iterator) {}

  // Storing a direct pointer to the OperatorDef even though we
  // already have the iterator saves an instruction in the critical
  // dispatch path. The iterator is effectively a
  OperatorDef* operator_def_;
  std::list<OperatorDef>::iterator operators_iterator_;
};

/**
 * This makes sure the template receives has more than 1 types
 */
template<class FuncType>
class TypedOperatorHandle final {
  static_assert(guts::false_t<FuncType>(),
    "FuncType in OperatorHandle::typed<FuncType> was not a valid function type");
};

class Dispatcher final {
public:
  Dispatcher():guard_(std::make_shared<Guard>()) {};

  struct Guard {
    Guard(): alive(true), mutex() {}
    std::atomic<bool> alive;
    std::mutex mutex;
  };

  friend class OperatorHandle;
  ~Dispatcher();

  template<class Return, class... Args>
  Return call(const TypedOperatorHandle<Return(Args...)>& op, Args... args) const;

  template<class Return, class... Args>
  Return redispatch(const TypedOperatorHandle<Return(Args...)>& op, DispatchKeySet dispatch_keyset, Args... args) const;

  std::optional<OperatorHandle> findSchema(const OperatorName& operator_name);

  OperatorHandle findSchemaOrThrow(const char* name, const char* overload_name);

  std::optional<OperatorHandle> findOp(const OperatorName& operator_name);

  static Dispatcher& singleton();

  RegistrationHandleRAII registerDef(FunctionSchema schema, std::string debug);

  RegistrationHandleRAII registerImpl(
    OperatorName op_name, DispatchKey dispatch_key, KernelFunction kernel_function,
    std::optional<CppSignature> cpp_signature, std::unique_ptr<FunctionSchema> inferred_function_schema,
    std::string debug);

private:
  std::list<OperatorDef> operators_;
  LeftRight<std::unordered_map<OperatorName, OperatorHandle, c10::OperatorNameHash>> operator_lookup_table_;
  void deregisterDef_(const OperatorHandle& op, const OperatorName& op_name);
  void deregisterImpl_(
    const OperatorHandle& op,
    const OperatorName& op_name,
    DispatchKey dispatch_key);
  
  OperatorHandle findOrRegisterName_(const OperatorName& op_name);

  void cleanup(const OperatorHandle& op, const OperatorName& op_name);

  // have one shared_ptr here because we might need this guard even the dispatcher is freed
  std::shared_ptr<Guard> guard_;
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

  Return call(Args... args) const {
    return c10::Dispatcher::singleton().call(*this, args...);
  }

  Return redispatch(DispatchKeySet currentDispatchKeySet, Args... args) const {
    return c10::Dispatcher::singleton().redispatch(
      *this, currentDispatchKeySet, args...);
  }
private:
  explicit TypedOperatorHandle(std::list<OperatorDef>::iterator operator_iterator)
    : OperatorHandle(operator_iterator) {}
};
}
