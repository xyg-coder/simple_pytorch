#pragma once

#include "Scalar.h"
#include "ScalarType.h"
#include "Tensor.h"
namespace simpletorch::impl {
Tensor tensor_fill(c10::Int64ArrayRef size, c10::ScalarType scalarType,
  const c10::Scalar& value);
} // namespace simpletorch::impl
