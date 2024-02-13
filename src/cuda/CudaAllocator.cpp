#include "cuda/CudaAllocator.h"
#include "Allocator.h"

namespace c10::cuda {
namespace cuda_allocator{

struct CudaMallocAsyncAllocator : public CUDAAllocator {
  void init(int dev_count) override {}
  DataPtr allocate(int64_t n) const override {
    return DataPtr();
  }
};

static CudaMallocAsyncAllocator cuda_malloc_async_allocator;

CUDAAllocator* get() {
  return &cuda_malloc_async_allocator;
}
}
}
