#pragma once

#include "Scalar.h"
#include "ScalarType.h"
#include "TensorIterator.h"
#include "cuda/Loops.cuh"
#include "utils/Exception.h"
namespace simpletorch::impl {

template<typename scalar_t>
struct FillFunctor {
  FillFunctor(scalar_t v): value(v) {}
  __device__ __forceinline__ scalar_t operator() () const {
    return value;
  }
  private:
    scalar_t value;
};



void fill_kernel_scalar_cuda(TensorIterator& iter, const c10::Scalar& value) {
#define DEFINE_SCALAR_TYPE_CASE(type, name) \
case c10::ScalarType::name: \
  c10::cuda::gpu_kernel(iter, FillFunctor<type>(value.to<type>()));

switch (iter.dtype()) {
  AT_FORALL_SCALAR_TYPES(DEFINE_SCALAR_TYPE_CASE)
  default:
    TORCH_CHECK(false, "The scalartype is not implemented");
}

#undef DEFINE_SCALAR_TYPE_CASE
}

} // namespace simpletorch::impl
