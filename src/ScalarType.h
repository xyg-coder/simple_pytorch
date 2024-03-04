#pragma once

#include <cstdint>
#include <ostream>

// with this way, we can define the way to iterate over the list
#define AT_FORALL_SCALAR_TYPES(_)    \
  _(uint8_t, Byte) /* 0 */           \
  _(int8_t, Char) /* 1 */            \
  _(int16_t, Short) /* 2 */          \
  _(int, Int) /* 3 */                \
  _(int64_t, Long) /* 4 */           \
  _(float, Float) /* 6 */            \
  _(double, Double) /* 7 */          \

namespace c10 {

enum class ScalarType : int8_t {
#define FETCH_SECOND_ELEMENT(f, s) s,
AT_FORALL_SCALAR_TYPES(FETCH_SECOND_ELEMENT)
#undef FETCH_SECOND_ELEMENT
  Undefined,
  NumOptions,
};

constexpr uint16_t NumScalarTypes =
  static_cast<uint16_t>(ScalarType::NumOptions);

static inline const char* toString(ScalarType t) {
#define DEFINE_CASE(_, name) \
  case ScalarType::name: \
    return #name;
  
  switch (t) {
    AT_FORALL_SCALAR_TYPES(DEFINE_CASE)
    default:
      return "UNKNOWN_CASE";
  }

#undef DEFINE_CASE
}

inline std::ostream& operator<<(std::ostream& stream, ScalarType scalar_type) {
  return stream << toString(scalar_type);
}
} // namespace c10
