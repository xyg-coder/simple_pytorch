#include "dispatch/Dispatcher.h"
#include "dispatch/OperatorName.h"
#include "utils/Exception.h"
#include <optional>

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

}
