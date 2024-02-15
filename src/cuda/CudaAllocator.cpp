#include "cuda/CudaAllocator.h"
#include "Allocator.h"
#include <cstddef>
#include <vector>

namespace c10::cuda {
namespace cuda_allocator{

int device_count;
std::vector<size_t> device_used_bytes;
std::vector<size_t> device_memory_limits;

struct CudaMallocAsyncAllocator : public CUDAAllocator {
  void init(int dev_count) override {
    static bool initialized = [](int dev_count){
      device_count = dev_count;
      device_memory_limits.resize(dev_count);
      device_used_bytes.resize(dev_count);
      return true;
    }(dev_count);
  }

  DataPtr allocate(int64_t n) const override {
    return DataPtr();
  }
private:
};

static CudaMallocAsyncAllocator cuda_malloc_async_allocator;

CUDAAllocator* get() {
  return &cuda_malloc_async_allocator;
}
}
}
