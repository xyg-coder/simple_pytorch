#include "cuda_ops/TensorFactories.h"
#include "Tensor.h"
#include "ops/EmptyOps.h"
#include "ops/FillOps.h"
#include <optional>

namespace simpletorch::impl {

Tensor tensor_fill(c10::Int64ArrayRef size, c10::ScalarType scalarType, const c10::Scalar& value) {
  Tensor tensor = simpletorch::ops::empty::call(size, scalarType,
    std::nullopt, std::nullopt);
  simpletorch::ops::fill::call(tensor, value);
  return tensor;
}

} // namespace simpletorch::impl
