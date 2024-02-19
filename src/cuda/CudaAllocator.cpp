#include "cuda/CudaAllocator.h"
#include "Allocator.h"
#include "Device.h"
#include "DeviceType.h"
#include "cuda/CUDAException.h"
#include "cuda/CUDAFunctions.h"
#include "cuda/CUDAStream.h"
#include "cuda/CudaGuard.h"
#include "utils/Exception.h"
#include "utils/Logging.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cuda_runtime.h>

namespace c10::cuda {

namespace {
struct UsageStream;

int device_count_;
std::vector<bool> devs_initialized_flags;
std::vector<UsageStream> dummy_unifying_free_streams;
std::vector<size_t> device_used_bytes;
std::vector<size_t> device_memory_limits;

// Possible micro-optimization:
// Some accesses to ptr_info are read-only.
// We could let those be concurrent with a shared_mutex and
// have concurrent calls take a shared_lock.
// Keeping it simple with an ordinary mutex for now.
std::mutex general_mutex;

struct UsageStream {
  cudaStream_t stream;
  c10::DeviceIndex device;
  UsageStream() = default;
  UsageStream(cudaStream_t s, c10::DeviceIndex d) : stream(s), device(d) {}
  UsageStream(const UsageStream& us) = default;
  UsageStream(const UsageStream&& us) : stream(us.stream), device(us.device) {}
  UsageStream& operator=(UsageStream other) {
    stream = other.stream;
    device = other.device;
    return *this;
  }
};

bool operator==(const UsageStream& lhs, const UsageStream& rhs) {
  return (lhs.stream == rhs.stream) && (lhs.device == rhs.device);
}

struct UsageStreamHash {
  size_t operator()(const UsageStream& us) const noexcept {
    return std::hash<void*>{}(us.stream) + size_t(us.device);
  }
};

struct PtrUsage {
  PtrUsage(uint64_t size): size(size) {}
  // recorded_streams doesn't include the original creation stream
  std::unordered_set<UsageStream, UsageStreamHash> recorded_streams;
  UsageStream creation_stream;
  uint64_t size;
};

using PtrInfo = std::unordered_map<void*, PtrUsage>;
PtrInfo ptr_info;


// assume the caller holds the mutex lock
void lazy_init_device(c10::DeviceIndex device) {
  if (devs_initialized_flags[device]) {
    return;
  }
  LOG_INFO("CudaMallocAsyncAllocator::lazy_init", device);
  CudaGuard g(device);
  cudaMemPool_t mempool = nullptr;

  C10_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&mempool, device));
  uint64_t threshold = UINT64_MAX;
  C10_CUDA_CHECK(cudaMemPoolSetAttribute(
    mempool, cudaMemPoolAttrReleaseThreshold, &threshold));
  
  // I think all these are on by default, but I want to enable them
  // explicitly to ensure awareness.
  int enable = 1;
  C10_CUDA_CHECK(cudaMemPoolSetAttribute(
    mempool, cudaMemPoolReuseFollowEventDependencies, &enable));
  C10_CUDA_CHECK(cudaMemPoolSetAttribute(
    mempool, cudaMemPoolReuseAllowOpportunistic, &enable));
  C10_CUDA_CHECK(cudaMemPoolSetAttribute(
    mempool, cudaMemPoolReuseAllowInternalDependencies, &enable));

  const auto dufs = getStreamFromPool();
  dummy_unifying_free_streams[device] = UsageStream(
    dufs.stream(), dufs.device_index());
  

  device_used_bytes[device] = 0;
  device_memory_limits[device] = UINT64_MAX;
  devs_initialized_flags[device] = true;
}

void mallocAsync(
  void **devPtr,
  c10::DeviceIndex device,
  size_t size,
  cudaStream_t stream) {
  // need to use (int)device otherwise it will be interpreted as char
  TORCH_CHECK(0 <= device && device < device_count_, "Invalid device, ", (int)device, ", device_count=", device_count_);
  CudaGuard g(device);
  std::lock_guard<std::mutex> lk(general_mutex);

  // TODO: original implementation includes capture checking here
  lazy_init_device(device);
  auto err = cudaGetLastError();
  C10_CUDA_CHECK(err);
  if (device_used_bytes[device] + size > device_memory_limits[device]) {
    err = cudaErrorMemoryAllocation; 
  } else {
    err = cudaMallocAsync(devPtr, size, stream);
  }

  if (err == cudaErrorMemoryAllocation) {
    // Clears CUDA's internal error state so the user, if desired, can catch the
    // OOM exception, free some stuff on the script side, and retry the
    // allocation. This aligns with the behavior of alloc_block in
    // CUDACachingAllocator.cpp.
    (void)cudaGetLastError(); // clear CUDA error
    TORCH_CHECK_WITH(OutOfMemoryError, false,
      "allocation on device ", device, " would exceed allowed memory\n");
  } else {
    C10_CUDA_CHECK(err);
  }

  // allocation succeeds
  auto inserted = ptr_info.emplace(*devPtr, PtrUsage(size));
  TORCH_CHECK(inserted.second, "address returned from cudaMallocAsync already exists");
  inserted.first->second.creation_stream = UsageStream(stream, device);
  device_used_bytes[device] += size;
}

inline void sync_raw(cudaStream_t dependency, cudaStream_t dependent) {
  // make sure the dependent streams will continue after current dependency stream is done
  cudaEvent_t event = nullptr;  
  C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  C10_CUDA_CHECK(cudaEventRecord(event, dependency));
  C10_CUDA_CHECK(cudaStreamWaitEvent(dependency, event));
  // any associated resources will automatically be released asynchronously at completion.
  C10_CUDA_CHECK(cudaEventDestroy(event));
}

inline void free_impl(PtrInfo::iterator& it) {
  LOG_INFO("trying to free cuda memory");
  const auto& recorded_streams = it->second.recorded_streams;
  const auto& creation_stream = it->second.creation_stream;

  // If the usage stream is a null (default) stream,
  // cudaFreeAsync infers the device from the ambient context,
  // so we need to set the right ambient context.
  CudaGuard g(creation_stream.device);

  // if no recording_stream, use creation stream to destroy
  if (it->second.recorded_streams.empty()) {
    C10_CUDA_CHECK(cudaFreeAsync(it->first, creation_stream.stream));
  } else {
    auto free_stream = dummy_unifying_free_streams[creation_stream.device];
    TORCH_CHECK(free_stream.device == creation_stream.device);
    sync_raw(creation_stream.stream, free_stream.stream);
    for (const auto& recorded_stream : recorded_streams) {
      // cudaEventRecord requires that the input event and stream are on the
      // same device.
      CudaGuard g(recorded_stream.device);
      sync_raw(recorded_stream.stream, free_stream.stream);
    }
    C10_CUDA_CHECK(cudaFreeAsync(it->first, free_stream.stream));
  }
  device_used_bytes[creation_stream.device] -= it->second.size;
  ptr_info.erase(it);
}

void freeAsync(void *ptr) {
  std::lock_guard<std::mutex>lock(general_mutex);
  C10_CUDA_CHECK(cudaGetLastError());
  auto it = ptr_info.find(ptr);
  TORCH_CHECK(it != ptr_info.end(), "ptr not found in ptr_info");
  free_impl(it);
}
}

namespace cuda_allocator{


struct CudaMallocAsyncAllocator : public CUDAAllocator {
  void init(int dev_count) override {
    static bool initialized = [](int dev_count){
      device_count_ = dev_count;
      device_memory_limits.resize(dev_count);
      devs_initialized_flags.resize(dev_count);
      device_used_bytes.resize(dev_count);
      dummy_unifying_free_streams.resize(dev_count);
      return true;
    }(dev_count);
  }

  DataPtr allocate(int64_t size) const override {
    constexpr size_t one_exa_bytes = 1152921504606846976ULL;
    TORCH_CHECK_WITH(
      OutOfMemoryError,
      size < one_exa_bytes,
      "CUDA out of memory. Tried to allocate more than 1EB memory.");
    c10::DeviceIndex device_index = 0;
    C10_CUDA_CHECK(c10::cuda::GetDevice(&device_index));
    void *r = nullptr;
    if (size != 0) {
      mallocAsync(&r, device_index, size, 
        c10::cuda::getCurrentCUDAStream(device_index).stream());
    }
    return DataPtr(r, r,
      &freeAsync, Device(DeviceType::CUDA, device_index));
  }
};

static CudaMallocAsyncAllocator cuda_malloc_async_allocator;

CUDAAllocator* get() {
  return &cuda_malloc_async_allocator;
}
}
}
