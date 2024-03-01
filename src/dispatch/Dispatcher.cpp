#include "dispatch/Dispatcher.h"
#include "dispatch/DispatchKeySet.h"
#include "dispatch/KernelFunction.h"
#include "dispatch/OperatorName.h"
#include "dispatch/RegistrationHandleRAII.h"
#include "utils/Exception.h"
#include <mutex>
#include <optional>
#include <unordered_map>

namespace c10 {
OperatorHandle Dispatcher::findSchemaOrThrow(
  const char* name, const char* overload_name) {
  const OperatorName operator_name(name, overload_name);
  auto it = findSchema(operator_name);
  if (!it.has_value()) {
    auto it2 = findOp(operator_name);
    if (it2.has_value()) {
      TORCH_CHECK(false,
        "Could not find schema for ", name, ".", overload_name,
        " but we found an implementation. Did you forget to def() the operator?");
    } else {
      TORCH_CHECK(false, "Could not find schema for ", name, ".", overload_name);
    }
  }
  return it.value();
}

Dispatcher::~Dispatcher() {
  std::lock_guard<std::mutex> lock(guard_->mutex);
  guard_->alive.store(false);
}

std::optional<OperatorHandle> Dispatcher::findSchema(const OperatorName& op_name) {
  auto it = findOp(op_name);
  if (it.has_value()) {
    if (it->hasSchema()) {
      return it.value();
    } else {
      return std::nullopt;
    }
  } else {
    return std::nullopt;
  }
}

std::optional<OperatorHandle> Dispatcher::findOp(const OperatorName& operator_name) {
  return operator_lookup_table_.read([&]
    (const std::unordered_map<OperatorName, OperatorHandle, OperatorNameHash>& operator_lookup_table) ->
      std::optional<OperatorHandle>{
      auto found = operator_lookup_table.find(operator_name);
      if (found == operator_lookup_table.end()) {
        return std::nullopt;
      }
      return found->second;
    }); 
}

Dispatcher& Dispatcher::singleton() {
  static Dispatcher _singleton;
  return _singleton;
}

template <class Return, class... Args>
Return Dispatcher::call(const TypedOperatorHandle<Return(Args...)>& op, Args... args) const {
  DispatchKeySet dispatch_keyset = op.operator_def_->op.dispatchKeyExtractor()
    .template getDispatchKeySetUnboxed<Args...>(args...);
  const KernelFunction& kernel = op.operator_def_->op.lookup(dispatch_keyset);
  return kernel.template call<Return, Args...>(
    op, dispatch_keyset, std::forward<Args>(args)...);
}

template <class Return, class... Args>
Return Dispatcher::redispatch(const TypedOperatorHandle<Return(Args...)>& op,
  DispatchKeySet dispatch_keyset, Args... args) const {
  const KernelFunction& kernel = op.operator_def_->op.lookup(dispatch_keyset);
    return kernel.template call<Return, Args...>(
    op, dispatch_keyset, std::forward<Args>(args)...);
}

OperatorHandle Dispatcher::findOrRegisterName_(const OperatorName& op_name) {
  const auto found = findOp(op_name);
  if (found != std::nullopt) {
    return *found;
  }

  operators_.emplace_back(std::move(OperatorName(op_name)));
  OperatorHandle handle(--operators_.end());
  operator_lookup_table_.write([&]
    (std::unordered_map<OperatorName, OperatorHandle, OperatorNameHash>& operator_lookup_table) {
      operator_lookup_table.emplace(op_name, handle);
    });
  return handle;
}

RegistrationHandleRAII Dispatcher::registerDef(FunctionSchema schema, std::string debug) {
  std::lock_guard<std::mutex> lock(guard_->mutex);
  OperatorName op_name = schema.operatorName();
  OperatorHandle op = findOrRegisterName_(op_name);
  TORCH_CHECK(op.operator_def_->def_count == 0,
    "Tried to register an operator (", schema, ") with the same name and overload name multiple times.");
  op.operator_def_->op_.registerSchema(std::move(schema), std::move(debug));

  ++op.operator_def_->def_count;
  ++op.operator_def_->def_and_impl_count;

  return RegistrationHandleRAII([guard=guard_, this, op, op_name]() -> void {
    std::lock_guard<std::mutex> lock(guard->mutex);
    if (!guard->alive.load()) {
      return;
    }
    deregisterDef_(op, op_name);
  });  
}

void Dispatcher::deregisterDef_(const OperatorHandle& op, const OperatorName& op_name) {
  TORCH_CHECK(op.schema().operatorName() == op_name);
  TORCH_CHECK(op.operator_def_->def_and_impl_count > 0);
  TORCH_CHECK(op.operator_def_->def_count > 0);
  --op.operator_def_->def_and_impl_count;
  --op.operator_def_->def_count;

  if (op.operator_def_->def_count == 0) {
    op.operator_def_->op_.deregisterSchema();
  }
  cleanup(op, op_name);
}

void Dispatcher::cleanup(const OperatorHandle& op, const OperatorName& op_name) {
  if (op.operator_def_->def_and_impl_count == 0) {
    operators_.erase(op.operators_iterator_);
    operator_lookup_table_.write(
      [&](std::unordered_map<OperatorName, OperatorHandle, c10::OperatorNameHash>& operator_lookup_table){
        operator_lookup_table.erase(op_name);
      });
  }
}

RegistrationHandleRAII Dispatcher::registerImpl(
  OperatorName op_name, DispatchKey dispatch_key, KernelFunction kernel_function,
  std::optional<CppSignature> cpp_signature, std::unique_ptr<FunctionSchema> inferred_function_schema,
  std::string debug) {
  std::lock_guard<std::mutex> lock(guard_->mutex);
  auto op = findOrRegisterName_(op_name);
  op.operator_def_->op_.registerKernel(
    *this,
    dispatch_key,
    std::move(kernel_function),
    std::move(cpp_signature),
    std::move(inferred_function_schema),
    std::move(debug));
  ++op.operator_def_->def_and_impl_count;
  return RegistrationHandleRAII([
    guard=this->guard_, this, op, op_name, dispatch_key]{
    std::lock_guard<std::mutex> lock(guard->mutex);
    if (guard->alive.load()) {
      return;
    }
    deregisterImpl_(op, op_name, dispatch_key);
  });
}

void Dispatcher::deregisterImpl_(const OperatorHandle& op,
  const OperatorName& op_name, DispatchKey dispatch_key) {
    
  op.operator_def_->op_.deregisterKernel_(*this, dispatch_key);
  TORCH_CHECK(op.operator_name() == op_name);
  TORCH_CHECK(op.operator_def_->def_and_impl_count > 0);
  --op.operator_def_->def_and_impl_count;
  cleanup(op, op_name);
}

} // namespace c10
