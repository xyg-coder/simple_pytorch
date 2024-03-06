#pragma once

#include "Device.h"
#include "MemoryFormat.h"
#include "ScalarType.h"
#include "Tensor.h"
#include "dispatch/DispatchKeySet.h"
#include "macros/Macros.h"
#include "utils/ArrayRef.h"
#include <optional>
namespace simpletorch {
namespace ops {

struct empty {
using schema = Tensor(c10::Int64ArrayRef, c10::ScalarType, std::optional<c10::Device>, std::optional<c10::MemoryFormat>);
STATIC_CONSTEXPR_STR(name, "empty");
STATIC_CONSTEXPR_STR(overload_name, "");
static Tensor call(
  c10::Int64ArrayRef size,
  c10::ScalarType scalarType,
  std::optional<c10::Device> device_opt,
  std::optional<c10::MemoryFormat> memory_format_opt);
static Tensor redispatch(
  c10::DispatchKeySet dispatchKetSet,
  c10::Int64ArrayRef size,
  c10::ScalarType scalarType,
  std::optional<c10::Device> device_opt,
  std::optional<c10::MemoryFormat> memory_format_opt
);
};

} // namespace ops
} // namespace simpletorch