#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <sys/types.h>
#include "cuda/CUDAStream.h"
#include "Device.h"
#include "Stream.h"
#include "cuda/CUDAException.h"
#include "cuda/CUDAFunctions.h"
#include "cuda/CudaGuard.h"
#include "utils/CallOnce.h"
#include "utils/Exception.h"

namespace c10::cuda {
namespace {
static c10::OnceFlag init_flag;
static DeviceIndex num_gpus = -1;
static int max_stream_priorities;
static constexpr int kStreamTypeBits = 4;

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
  int leastPriority = -1, greatestPriority = -1;
  C10_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
  auto range = leastPriority - greatestPriority + 1;
  max_stream_priorities = range >= c10::cuda::max_compile_time_stream_priorities
    ? c10::cuda::max_compile_time_stream_priorities
    : range;
}

static void initCUDAStreamsOnce() {
  c10::callOnce(init_flag, initCUDAStreamsOnce);

  if (current_stream) {
    return;
  }

  current_stream = std::make_unique<StreamId[]>(num_gpus);
  for (auto i = 0; i < num_gpus; ++i) {
    current_stream[i] = makeStreamId(StreamIdType::DEFAULT, 0);
  }
}

static void initDeviceStreamState(DeviceIndex device_index) {
  // Switches to the requested device so streams are properly associated
  // with it.
  CudaGuard device_guard(device_index);
}

CUDAStream getStreamFromPool(const int priority, DeviceIndex device_index) {
  initCUDAStreamsOnce();

}
}
}
