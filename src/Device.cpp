#include "Device.h"
#include "DeviceType.h"
#include <glog/logging.h>
#include <stdexcept>

namespace c10 {
std::string Device::str() const {
  std::string str = DeviceTypeName(type_, /* lower case */ true);
  if (has_index()) {
    str.push_back(':');
    str.append(std::to_string(index_));
  }
  return str;
}

std::ostream& operator<<(std::ostream& stream, const Device& device) {
  stream << device.str();
  return stream;
}

void Device::validate() {
  if (index_ < -1) {
    throw std::invalid_argument("Device index must be -1 or non negative, but got " + std::to_string(index_));
  } 
  if (type_ == DeviceType::CPU && (index_ != 0 && index_ != -1)) {
    throw std::invalid_argument("Device index must be -1 or non negative, but got " + std::to_string(index_));
  }
}
}
