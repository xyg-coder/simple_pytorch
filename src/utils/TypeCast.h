#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utils/Exception.h>
namespace c10 {

/// Returns true if x is greater than the greatest value of the type Limit
template <typename Limit, typename T>
inline constexpr bool greater_than_max(const T& x) {
  constexpr bool can_overflow =
      std::numeric_limits<T>::digits > std::numeric_limits<Limit>::digits;
  return can_overflow && x > std::numeric_limits<Limit>::max();
}

/// Returns false since we cannot have x < 0 if x is unsigned.
template <typename T>
static inline constexpr bool is_negative(
    const T& /*x*/,
    std::true_type /*is_unsigned*/) {
  return false;
}

/// Returns true if a signed variable x < 0
template <typename T>
static inline constexpr bool is_negative(
    const T& x,
    std::false_type /*is_unsigned*/) {
  return x < T(0);
}

/// Returns true if x < 0
/// NOTE: Will fail on an unsigned custom type
///       For the most part it's possible to fix this if
///       the custom type has a constexpr constructor.
///       However, notably, c10::Half does not :-(
template <typename T>
inline constexpr bool is_negative(const T& x) {
  return is_negative(x, std::is_unsigned<T>());
}

/// Returns true if x < lowest(Limit). Standard comparison
template <typename Limit, typename T>
static inline constexpr bool less_than_lowest(
    const T& x,
    std::false_type /*limit_is_unsigned*/,
    std::false_type /*x_is_unsigned*/) {
  return x < std::numeric_limits<Limit>::lowest();
}

/// Returns false since all the limit is signed and therefore includes
/// negative values but x cannot be negative because it is unsigned
template <typename Limit, typename T>
static inline constexpr bool less_than_lowest(
    const T& /*x*/,
    std::false_type /*limit_is_unsigned*/,
    std::true_type /*x_is_unsigned*/) {
  return false;
}

/// Returns true if x < 0, where 0 is constructed from T.
/// Limit is not signed, so its lower value is zero
template <typename Limit, typename T>
static inline constexpr bool less_than_lowest(
    const T& x,
    std::true_type /*limit_is_unsigned*/,
    std::false_type /*x_is_unsigned*/) {
  return x < T(0);
}

/// Returns false sign both types are unsigned
template <typename Limit, typename T>
static inline constexpr bool less_than_lowest(
    const T& /*x*/,
    std::true_type /*limit_is_unsigned*/,
    std::true_type /*x_is_unsigned*/) {
  return false;
}

/// Returns true if x is less than the lowest value of type T
/// NOTE: Will fail on an unsigned custom type
///       For the most part it's possible to fix this if
///       the custom type has a constexpr constructor.
///       However, notably, c10::Half does not :
template <typename Limit, typename T>
inline constexpr bool less_than_lowest(const T& x) {
  return less_than_lowest<Limit>(
      x, std::is_unsigned<Limit>(), std::is_unsigned<T>());
}

// bool can be converted to any type.
// Without specializing on bool, in pytorch_linux_trusty_py2_7_9_build:
// `error: comparison of constant '255' with boolean expression is always false`
// for `f > limit::max()` below
template <typename To, typename From>
std::enable_if_t<std::is_same_v<From, bool>, bool> overflows(
    From /*f*/,
    bool strict_unsigned = false) {
  return false;
}

// skip isnan and isinf check for integral types
template <typename To, typename From>
std::enable_if_t<std::is_integral_v<From> && !std::is_same_v<From, bool>, bool>
overflows(From f, bool strict_unsigned = false) {
  using limit = std::numeric_limits<To>;
  if constexpr (!limit::is_signed && std::numeric_limits<From>::is_signed) {
    if (!strict_unsigned) {
      return greater_than_max<To>(f) || (c10::is_negative(f) && -static_cast<uint64_t>(f) > static_cast<uint64_t>(limit::max()));
    }
  }

  return c10::less_than_lowest<To>(f) || greater_than_max<To>(f);
}

template <typename To, typename From>
std::enable_if_t<std::is_floating_point_v<From>, bool> overflows(
    From f,
    bool strict_unsigned = false) {
  using limit = std::numeric_limits<To>;
  if (limit::has_infinity && std::isinf(static_cast<double>(f))) {
    return false;
  }
  if (!limit::has_quiet_NaN && (f != f)) {
    return true;
  }
  return f < limit::lowest() || f > limit::max();
}

template <typename To, typename From>
To checked_convert(From f, const char* name) {
  if (!std::is_same_v<To, bool> && overflows<To, From>(f)) {
    TORCH_CHECK(false, "value cannot be converted to type ", name, " without overflow");
  }
  return static_cast<To>(f);
}
};
