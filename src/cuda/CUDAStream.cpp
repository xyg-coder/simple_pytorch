#include <atomic>
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <sys/types.h>
#include "cuda/CUDAStream.h"
#include "Device.h"
#include "DeviceType.h"
#include "Stream.h"
#include "cuda/CUDAException.h"
#include "cuda/CUDAFunctions.h"
#include "cuda/CudaGuard.h"
#include "macros/CUDAMacros.h"
#include "utils/CallOnce.h"
#include "utils/Exception.h"
#include "utils/Logging.h"

namespace c10::cuda {
namespace {
static c10::OnceFlag init_flag;
static DeviceIndex num_gpus = -1;
static int least_priority = -1;
// this highest_priority has already capped by max_compile_time_stream_priorities
static int highest_priority = -1;
static int max_stream_priorities;
static constexpr int kStreamsPerPoolBits = 5;
static constexpr int kStreamsPerPool = 1 << kStreamsPerPoolBits;
static constexpr int kStreamTypeBits = 4;
static constexpr unsigned int kDefaultFlags = cudaStreamNonBlocking;

static c10::OnceFlag device_flags[C10_COMPILE_TIME_MAX_GPUS];
static cudaStream_t streams[max_compile_time_stream_priorities]
  [C10_COMPILE_TIME_MAX_GPUS][kStreamsPerPool];

static std::atomic<uint32_t> priority_counters[max_compile_time_stream_priorities]
  [C10_COMPILE_TIME_MAX_GPUS];

// Thread-local current streams
static thread_local std::unique_ptr<StreamId[]> current_stream = nullptr;

// Non-default streams
// Note: the number of CUDA devices is determined at run time,
// and the low and high priority pools are lazily initialized
// when the first stream is requested for a device.
// The device flags track the initialization of each device, while
// the low and high priority counters track, for each device, the next stream
// in the pool to be returned when a stream is requested (round-robin fashion
// , see the note in CUDAStream.h).
// The streams are "leaked": they are created but never destroyed because the
// destruction of global variables could happen after the CUDA runtime has
// already been destroyed and thus invoking cudaStreamDestroy could lead to a
// crash. It's likely an issue in CUDA, but to be safe - let's just "forget"
// the destruction.

// Note [StreamId assignment]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// How do we assign stream IDs?
//
// -- 54 bits --  -- 5 bits -----  -- 4 bits --     --1 bit --
// zeros          stream id index  StreamIdType     Ext/native stream
//                ignored for ext   ignored for ext
// for external stream, StreamID is a cudaStream_t pointer
// this means that last bit will always be 0
// so when constructing StreamId for a native stream we set last bit to 1
// to distinguish between native and external streams
//
//
// We are obligated to treat the stream ID 0 as the default stream, per the
// invariant specified in c10::Stream, so this is one exception to
// "last bit = 1 for native streams". However, all other numbers are entirely
// an internal implementation detail, we reserve the right to renumber streams
// however we like.
//
// Note that it is really important that the MSB is zero; StreamId is a
// *signed* integer, and unsigned to signed conversion outside of the
// bounds of signed integer representation is undefined behavior.  You
// could work around this with something like
// https://stackoverflow.com/questions/13150449/efficient-unsigned-to-signed-cast-avoiding-implementation-defined-behavior
// but it seems a bit overkill for this.
//
// Also, external managed stream pointers (cudaStream_t) can be directly stored
// in the Id field so in this case, we need to check the stream alignment.
class StreamIdType {
  // StreamIdType encodes whether this stream is DEFAULT, EXTernal or
  // for all other native streams, the stream priority (higher value is higher
  // priority)
private:
  uint8_t stream_type;
public:
  static const uint8_t DEFAULT = 0x0;
  static const uint8_t EXTERNAL = 0xF;

  StreamIdType(const uint8_t _stream_type) : stream_type(_stream_type) {}

  bool isExternal() const {
    return EXTERNAL == stream_type;
  }

  bool isDefault() const {
    return DEFAULT == stream_type;
  }

  uint8_t getStreamType() const {
    return stream_type;
  }
};

StreamId makeStreamId(StreamIdType st, size_t si) {
  if (st.isDefault()) {
    return static_cast<StreamId>(0);
  }

  return (static_cast<StreamId>(si) << (kStreamTypeBits + 1)) |
      static_cast<StreamId>(st.getStreamType() << 1) | 1;
}

std::ostream& operator<<(std::ostream& stream, StreamIdType s) {
  if (s.isDefault()) {
    stream << "DEFAULT";
  } else if (s.isExternal()) {
    stream << "EXTERNAL";
  } else {
    stream << "PRIORITY " << int(s.getStreamType());
  }
  return stream;
}

static void initGlobalStreamState() {
  num_gpus = c10::cuda::device_count();
  TORCH_CHECK(num_gpus > 0, "There is no gpu found");
  C10_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &highest_priority));
  auto range = least_priority - highest_priority + 1;
  max_stream_priorities = range >= c10::cuda::max_compile_time_stream_priorities
    ? c10::cuda::max_compile_time_stream_priorities
    : range;
  highest_priority = least_priority - range + 1;
  LOG_INFO("cuda stream building, least priority: ", least_priority,
    ", highest priority", highest_priority);
}

static void initCUDAStreamsOnce() {
  c10::callOnce(init_flag, initGlobalStreamState);

  if (current_stream) {
    return;
  }

  current_stream = std::make_unique<StreamId[]>(num_gpus);
  for (auto i = 0; i < num_gpus; ++i) {
    current_stream[i] = makeStreamId(StreamIdType::DEFAULT, 0);
  }
}


inline int priority_to_idx(int priority) {
  int index = least_priority - priority;
  return std::min(index, max_stream_priorities);
}

inline int idx_to_priority(int idx) {
  return least_priority - idx;
}

static void initDeviceStreamState(DeviceIndex device_index) {
  LOG_INFO("initDeviceStreamState, creating device streams");
  // Switches to the requested device so streams are properly associated
  // with it.
  // CUDA apis are thread-safe, each threads are setting thread local device
  CudaGuard device_guard(device_index);
  for (auto i = 0; i < kStreamsPerPool; ++i) {
    for (auto j = 0; j < max_stream_priorities; ++j) {
      auto& stream = streams[j][device_index][i];
      // lower number is higher priority
      C10_CUDA_CHECK(cudaStreamCreateWithPriority(
        &stream, kDefaultFlags, idx_to_priority(j)));
      
      // initialize
      priority_counters[j][device_index] = 0;
    }
  }
  LOG_INFO("initDeviceStreamState, creating device streams done");
}

// Helper to verify the GPU index is valid
static inline void check_gpu(DeviceIndex device_index) {
  TORCH_CHECK(device_index >= 0 && device_index < num_gpus);
}

// Helper to determine the index of the stream to return
// Note: Streams are returned round-robin (see note in CUDAStream.h)
static uint32_t get_idx(std::atomic<uint32_t>& counter) {
  auto raw_idx = counter++;
  return raw_idx % kStreamsPerPool;
}

CUDAStream cudaStreamForId(DeviceIndex device_index, StreamId stream_id) {
  return CUDAStream(CUDAStream::UNCHECKED, 
    c10::Stream(c10::Stream::UNSAFE, 
    c10::Device(DeviceType::CUDA, device_index), stream_id));
}


}

CUDAStream getStreamFromPool(const int priority, DeviceIndex device_index) {
  initCUDAStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
    c10::cuda::SetTargetDevice();
  }
  TORCH_CHECK(priority <= 0, "Expected cuda stream priority to be less than or equal to 0, got ", priority);
  check_gpu(device_index);

  c10::callOnce(device_flags[device_index], initDeviceStreamState, device_index);

  auto pri_idx = priority_to_idx(priority);
  pri_idx =
      std::min(pri_idx, max_stream_priorities - 1); // pri_idx is zero-based
  const auto idx = get_idx(priority_counters[pri_idx][device_index]);
  StreamIdType id_type = StreamIdType(pri_idx + 1);
  return cudaStreamForId(device_index, makeStreamId(id_type, idx));
}

CUDAStream getStreamFromPool(const bool isHighPriority, DeviceIndex device) {
  initCUDAStreamsOnce();
  int priority = isHighPriority ? highest_priority : least_priority;
  return getStreamFromPool(priority, device);
}
}