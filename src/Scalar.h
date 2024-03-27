#pragma once

#include <cstdint>
#include <type_traits>
#include <ScalarType.h>
#include <utils/TypeCast.h>
namespace c10 {

/**
 * Scalar represents a 0-dimensional tensor which contains a single element.
 * Unlike a tensor, numeric literals (in C++) are implicitly convertible to
 * Scalar (which is why, for example, we provide both add(Tensor) and
 * add(Scalar) overloads for many operations). It may also be used in
 * circumstances where you statically know a tensor is 0-dim and single size,
 * but don't know its type.
 */
class Scalar {
public:
// a macros that can create constructor for different types
#define DEFINE_IMPLICIT_CTOR(type, name) \
  Scalar(type vv) : Scalar(vv, true) {}

  AT_FORALL_SCALAR_TYPES(DEFINE_IMPLICIT_CTOR);

  Scalar(uint16_t vv) : Scalar(vv, true) {}
  Scalar(uint32_t vv) : Scalar(vv, true) {}
  Scalar(uint64_t vv) {
    if (vv > static_cast<uint64_t>(INT64_MAX)) {
      tag = Tag::HAS_u;
      v.u = vv;
    } else {
      tag = Tag::HAS_i;
      v.i = static_cast<int64_t>(vv);
    }
  }

  // also support scalar.to<int64_t>();
  // Deleted for unsupported types, but specialized below for supported types
  template <typename T>
  T to() const = delete;

private:
  enum class Tag {HAS_d, HAS_i, HAS_u};
  union v_t {
    double d{};
    int64_t i;
    uint64_t u;
    v_t() {}; // default constructor
  } v;
  Tag tag;


  template <
    typename T,
    typename std::enable_if_t<
        std::is_integral_v<T> && !std::is_same_v<T, bool>,
        bool>* = nullptr>
  Scalar(T vv, bool) : tag(Tag::HAS_i) {
    v.i = static_cast<int64_t>(vv);
  }

  template <
    typename T,
    typename std::enable_if_t<
        !std::is_integral_v<T>, bool>* = nullptr>
  Scalar(T vv, bool) : tag(Tag::HAS_d) {
    v.d = static_cast<double>(vv);
  }

#define DEFINE_ACCESSOR(type, name)                   \
type to##name() const {                               \
if (Tag::HAS_d == tag) {                              \
return checked_convert<type, double>(v.d, #type);     \
} else if (Tag::HAS_i == tag) {                       \
  return checked_convert<type, int>(v.i, #type);      \
} else {                                              \
return checked_convert<type, uint64_t>(v.u, #type);   \
}                                                     \
}

AT_FORALL_SCALAR_TYPES(DEFINE_ACCESSOR)
DEFINE_ACCESSOR(uint16_t, UInt16)
DEFINE_ACCESSOR(uint32_t, UInt32)
DEFINE_ACCESSOR(uint64_t, UInt64)

#undef DEFINE_ACCESSOR
};

#define DEFINE_TO(T, name)          \
template <>                         \
inline T Scalar::to<T>() const {    \
return to##name();                  \
}

AT_FORALL_SCALAR_TYPES(DEFINE_TO)
DEFINE_TO(uint16_t, UInt16)
DEFINE_TO(uint32_t, UInt32)
DEFINE_TO(uint64_t, UInt64)

#undef DEFINE_TO

} // namespace c10
