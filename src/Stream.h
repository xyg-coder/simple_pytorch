#pragma once

#include "Device.h"
#include <cstdint>
namespace c10 {
using StreamId = int64_t;

class Stream final {
private:
  Device device_;
  StreamId id_;
 public:
  enum Unsafe { UNSAFE };
  enum Default { DEFAULT };

    explicit Stream(Unsafe, Device device, StreamId id)
      : device_(device), id_(id) {}

  /// Construct the default stream of a Device.  The default stream is
  /// NOT the same as the current stream; default stream is a fixed stream
  /// that never changes, whereas the current stream may be changed by
  /// StreamGuard.
  explicit Stream(Default, Device device) : device_(device), id_(0) {}

  bool operator==(const Stream& other) const noexcept {
    return this->device_ == other.device_ && this->id_ == other.id_;
  }
  bool operator!=(const Stream& other) const noexcept {
    return !(*this == other);
  }

  Device device() const noexcept {
    return device_;
  }
  DeviceType device_type() const noexcept {
    return device_.type();
  }
  DeviceIndex device_index() const noexcept {
    return device_.index();
  }
  StreamId id() const noexcept {
    return id_;
  }
};

std::ostream& operator<<(std::ostream& stream, const Stream& s);
};
