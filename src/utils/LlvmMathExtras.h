#pragma once

#include <cstddef>
#include <limits>
namespace c10::llvm {


template <typename T, std::size_t SizeOfT>
struct LeadingZerosCounter {
  static std::size_t count(T Val) {
    if (!Val)
      return std::numeric_limits<T>::digits;

    // Bisection method.
    std::size_t ZeroBits = 0;
    for (T Shift = std::numeric_limits<T>::digits >> 1; Shift; Shift >>= 1) {
      T Tmp = Val >> Shift;
      if (Tmp)
        Val = Tmp;
      else
        ZeroBits |= Shift;
    }
    return ZeroBits;
  }
};

/// Count number of 0's from the most significant bit to the least
///   stopping at the first 1.
///
/// Only unsigned integral types are allowed.
///
/// \param ZB the behavior on an input of 0. Only ZB_Width and ZB_Undefined are
///   valid arguments.
template <typename T>
std::size_t countLeadingZeros(T Val) {
  static_assert(
      std::numeric_limits<T>::is_integer && !std::numeric_limits<T>::is_signed,
      "Only unsigned integral types are allowed.");
  return LeadingZerosCounter<T, sizeof(T)>::count(Val);
}
}
