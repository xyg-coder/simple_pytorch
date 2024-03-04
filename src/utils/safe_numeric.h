#pragma once

#include <cstdint>
namespace c10 {

inline bool mul_overflows(uint64_t a, uint64_t b, uint64_t*out) {
  return __builtin_mul_overflow(a, b, out);
}

inline bool add_overflows(uint64_t a, uint64_t b, uint64_t* out) {
  return __builtin_add_overflow(a, b, out);
}

template <typename iterator>
bool safe_multiple_uid(iterator first, iterator last, uint64_t*out) {
  uint64_t prod = 1;
  bool overflow = false;
  for (auto i = first; i != last; ++i) {
    overflow |= mul_overflows(prod, *i, &prod);
  }
  *out = prod;
  return overflow;
}
}
