#pragma once

#include "Scalar.h"
#include "Tensor.h"
#include "dispatch/DispatchKeySet.h"
namespace simpletorch::ops {
struct fill {
using schema = Tensor&(Tensor& self, const c10::Scalar& value);
STATIC_CONSTEXPR_STR(name, "fill");
STATIC_CONSTEXPR_STR(overload_name, "");

static Tensor& call(Tensor& self, const c10::Scalar& value);
static Tensor& redispatch(c10::DispatchKeySet, Tensor& self, const c10::Scalar& value);
};
} // namespace simpletorch::ops
