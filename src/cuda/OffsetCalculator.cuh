#pragma once

#include "macros/Macros.h"
#include "utils/Array.h"
#include <algorithm>
namespace c10::cuda {
template <int NARGS, typename index_t = uint32_t>
struct TrivialOffsetCalculator {
  // The offset for each argument. Wrapper around fixed-size array.
  // The offsets are in # of elements, not in bytes.
  // On CUDA, zero sized array is not allowed, so when we are handling nullary
  // operators, we need to create a size 1 offset to avoid compiler failure.
  // This size 1 offset is just a placeholder, and we will not use it.

  using offset_type = c10::Array<index_t, std::max<int>(NARGS, 1)>;

  C10_HOST_DEVICE offset_type get(index_t linear_idx) {
    offset_type offsets;
    #pragma unroll
    for (int arg = 0; arg < NARGS; arg++) {
      offsets[arg] = linear_idx;
    }
    return offsets;
  }
};
}// namespace c10::cuda
