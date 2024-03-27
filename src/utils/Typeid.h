#pragma once

#include "ScalarType.h"
#include "utils/Exception.h"
#include <array>
#include <cstddef>
#include <cstdint>
namespace c10 {
class TypeMeta final {

// array of the sizeof types
static constexpr std::array<uint8_t, NumScalarTypes> scalarTypeItemSizes = {
#define MAP_TYPE_TO_SIZE(T, name) sizeof(T),
  AT_FORALL_SCALAR_TYPES(MAP_TYPE_TO_SIZE)
#undef MAP_TYPE_TO_SIZE
0, // undefined
};

public:
  TypeMeta() = delete;
  TypeMeta(uint16_t index): index_(index) {}
  
  static inline TypeMeta fromScalarType(ScalarType scalar_type) {
    const auto index = static_cast<uint16_t>(scalar_type);
    TORCH_CHECK(index < NumScalarTypes,
      "Unrecognized scalartype ", scalar_type);
    return TypeMeta(index);
  }

  size_t itemsize() const noexcept {
    return scalarTypeItemSizes[index_];
  }

  inline ScalarType toScalarType() const {
    return static_cast<ScalarType>(index_);
  }

private:
  uint16_t index_;

};
}
