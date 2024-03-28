#pragma once

#include "Device.h"
#include "MemoryFormat.h"
#include "ScalarType.h"
#include "Tensor.h"
#include "utils/ArrayRef.h"
#include <optional>

namespace simpletorch::impl {

Tensor empty_cuda(
  c10::Int64ArrayRef size,
  c10::ScalarType scalarType,
  std::optional<c10::Device> device_opt,
  std::optional<c10::MemoryFormat> memory_format_opt);

} // namespace simpletorch::impl
