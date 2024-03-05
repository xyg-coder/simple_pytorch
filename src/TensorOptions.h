#include "Device.h"
#include "DeviceType.h"
#include "MemoryFormat.h"
#include <optional>

#define VALUE_OR_ELSE(OPT, OTHERWISE) \
  if (OPT.has_value()) {              \
  return *OPT;                        \
  }                                   \
  return OTHERWISE

namespace simpletorch {

inline c10::Device device_or_default(std::optional<c10::Device> device_opt) {
  VALUE_OR_ELSE(device_opt, c10::Device(c10::DeviceType::CUDA));
}

inline c10::MemoryFormat memory_format_or_default(
  std::optional<c10::MemoryFormat> memory_format_opt) {

  VALUE_OR_ELSE(memory_format_opt, c10::MemoryFormat::Contiguous);
}

} // namespace simpletorch
