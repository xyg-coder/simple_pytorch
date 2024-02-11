#pragma once

#include <cstdint>
#include <string>

namespace c10 {
enum class DeviceType: int8_t {
    CPU = 0,
    CUDA = 1,
};

std::string DeviceTypeName(DeviceType d, bool lower_case = false);

std::ostream& operator<<(std::ostream& stream, DeviceType type);

}

namespace std {
template <>
struct hash<c10::DeviceType> {
  std::size_t operator()(c10::DeviceType k) const {
    return std::hash<int>()(static_cast<int>(k));
  }
};
} // namespace std
