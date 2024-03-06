#include "dispatch/Dispatcher.h"
#include "ops/EmptyOps.h"

namespace simpletorch::ops {

static c10::TypedOperatorHandle<empty::schema> create_zeros_typed_handle() {
  return c10::Dispatcher::singleton()
  .findSchemaOrThrow(empty::name, empty::overload_name)
  .typed<empty::schema>();
}

Tensor empty::call(c10::Int64ArrayRef size, c10::ScalarType scalarType, std::optional<c10::Device> device_opt, std::optional<c10::MemoryFormat> memory_format_opt) {
  static auto op = create_zeros_typed_handle();
  return op.call(size, scalarType, device_opt, memory_format_opt);
}

Tensor empty::redispatch(c10::DispatchKeySet dispatchKetSet, c10::Int64ArrayRef size, c10::ScalarType scalarType, std::optional<c10::Device> device_opt, std::optional<c10::MemoryFormat> memory_format_opt) {
  static auto op = create_zeros_typed_handle();
  return op.redispatch(dispatchKetSet, size, scalarType, device_opt, memory_format_opt);
}

}
