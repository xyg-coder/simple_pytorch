#include "cuda/CudaAllocator.h"
#include "Allocator.h"
#include "Device.h"
#include "cuda/CUDAException.h"
#include "cuda/CUDAStream.h"
#include "cuda/CudaGuard.h"
#include "utils/Exception.h"
#include "utils/Logging.h"
#include <cstddef>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

namespace c10::cuda {

namespace {
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
}

namespace cuda_allocator{

int device_count;
std::vector<bool> devs_initialized_flags;
std::vector<UsageStream> dummy_unifying_free_streams;
std::vector<size_t> device_used_bytes;
std::vector<size_t> device_memory_limits;

struct CudaMallocAsyncAllocator : public CUDAAllocator {

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

  void init(int dev_count) override {
    static bool initialized = [](int dev_count){
      device_count = dev_count;
      device_memory_limits.resize(dev_count);
      device_used_bytes.resize(dev_count);
      return true;
    }(dev_count);
  }

  DataPtr allocate(int64_t size) const override {
    constexpr size_t one_exa_bytes = 1152921504606846976ULL;
    TORCH_CHECK_WITH(
      OutOfMemoryError,
      size < one_exa_bytes,
      "CUDA out of memory. Tried to allocate more than 1EB memory.");
  }


};

static CudaMallocAsyncAllocator cuda_malloc_async_allocator;

CUDAAllocator* get() {
  return &cuda_malloc_async_allocator;
}
}
}
