#pragma once

#include <cstdint>
namespace c10 {
enum class DispatchKey : uint16_t {
  CUDA,
  EndOfFunctionalityKeys,
};
constexpr uint8_t num_functionality_keys = static_cast<uint8_t>(DispatchKey::EndOfFunctionalityKeys);

// compared to pytorch, we ignore the num_backends
// might add in the future
constexpr uint16_t num_runtime_entries = num_functionality_keys;
}
