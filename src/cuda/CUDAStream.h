# pragma once

#include "Device.h"
#include "DeviceType.h"
#include "Stream.h"
#include "utils/Exception.h"
#include <cuda_runtime.h>
namespace c10::cuda {
static constexpr int max_compile_time_stream_priorities = 4;

// Value object representing a CUDA stream.  This is just a wrapper
// around c10::Stream, but it comes with a little extra CUDA-specific
// functionality (conversion to cudaStream_t), and a guarantee that
// the wrapped c10::Stream really is a CUDA stream.
class CUDAStream {
public:
  enum Unchecked { UNCHECKED };
  explicit CUDAStream(Stream stream): stream_(stream) {
    TORCH_CHECK(stream_.device_type() == DeviceType::CUDA);
  }
  explicit CUDAStream(Unchecked, Stream stream): stream_(stream) {}

  bool operator==(const CUDAStream& other) const noexcept {
    return unwrap() == other.unwrap();
  }

  bool operator!=(const CUDAStream& other) const noexcept {
    return unwrap() != other.unwrap();
  }

  /// Explicit conversion to cudaStream_t.
  cudaStream_t stream() const;

  /// Explicit conversion to Stream.
  Stream unwrap() const {
    return stream_;
  }

  /// Get the CUDA device index that this stream is associated with.
  DeviceIndex device_index() const {
    return stream_.device_index();
  }
private:
  Stream stream_;
};

CUDAStream getStreamFromPool(const int priority, DeviceIndex device = -1);
CUDAStream getStreamFromPool(const bool isHighPriority = false, DeviceIndex device = -1);

CUDAStream getCurrentCUDAStream(DeviceIndex device_index = -1);
};
