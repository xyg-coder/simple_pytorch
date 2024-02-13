#pragma once

#include "Allocator.h"
namespace c10::cuda {
namespace cuda_allocator{
class CUDAAllocator : public Allocator {
public:
  virtual void init(int device_count) = 0;
};

CUDAAllocator* get();
}
}
