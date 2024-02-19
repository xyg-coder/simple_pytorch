#include "Allocator.h"
#include "Context.h"
#include "cuda/CudaAllocator.h"
#include "utils/Logging.h"
#include <glog/logging.h>

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  simpletorch::globalContext().lazyInitCUDA();
  c10::cuda::cuda_allocator::CUDAAllocator *allocator = c10::cuda::cuda_allocator::get();
  c10::DataPtr ptr = allocator->allocate(64 * 64 * sizeof(int));
  LOG_INFO("finish allocating");
}
