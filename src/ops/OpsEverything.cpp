#include "Tensor.h"
#include "dispatch/Dispatcher.h"
#include "ops/EmptyOps.h"
#include "ops/FillOps.h"

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

static c10::TypedOperatorHandle<fill::schema> create_fill_typed_handle() {
  return c10::Dispatcher::singleton()
  .findSchemaOrThrow(fill::name, fill::overload_name)
  .typed<fill::schema>();
}

Tensor& fill::call(Tensor& self, const c10::Scalar& value) {
  static auto op = create_fill_typed_handle();
  return op.call(self, value);
}

Tensor& fill::redispatch(c10::DispatchKeySet dispatchKetSet, Tensor& self, const c10::Scalar& value) {
  static auto op = create_fill_typed_handle();
  return op.redispatch(dispatchKetSet, self, value);
}

static c10::TypedOperatorHandle<tensor_fill::schema> create_tensor_fill_typed_handle() {
  return c10::Dispatcher::singleton()
  .findSchemaOrThrow(tensor_fill::name, tensor_fill::overload_name)
  .typed<tensor_fill::schema>();
}

Tensor tensor_fill::call(c10::Int64ArrayRef size, c10::ScalarType scalarType, const c10::Scalar &value) {
  static auto op = create_tensor_fill_typed_handle();
  return op.call(size, scalarType, value);
}

Tensor tensor_fill::redispatch(c10::DispatchKeySet dispatchKetSet, c10::Int64ArrayRef size, c10::ScalarType scalarType, const c10::Scalar &value) {
  static auto op = create_tensor_fill_typed_handle();
  return op.redispatch(dispatchKetSet, size, scalarType, value);
}

}
