#pragma once

#include "DeviceType.h"
#include <cstdint>
#include <ostream>
#include <string>

namespace c10 {
using DeviceIndex = int8_t;

struct Device final {
	Device(DeviceType device_type, DeviceIndex index=-1): type_(device_type), index_(index) {
			validate();
	}

	DeviceType type() const noexcept {
		return type_;
	}

  DeviceIndex index() const noexcept {
    return index_;
  }

	bool operator==(const Device& other) const noexcept {
			return this->type_ == other.type_ && this->index_ == other.index_;
	}

	bool operator!=(const Device& other) const noexcept {
			return !(*this == other);
	}

	bool is_cuda() const noexcept {
			return type_ == DeviceType::CUDA;
	}

	bool is_cpu() const noexcept {
			return type_ == DeviceType::CPU;
	}

	bool has_index() const noexcept {
			return index_ != -1;
	}

	// Same string as returned from operator<<.
	std::string str() const;

private:
	DeviceType type_;
	DeviceIndex index_ = -1;
	void validate();
};

std::ostream& operator<<(std::ostream& stream, const Device& device);
}
