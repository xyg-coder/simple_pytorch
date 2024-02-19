#include "Device.h"
#include <gtest/gtest.h>
#include "Allocator.h"
#include "Context.h"
#include "Stream.h"
#include "cuda/CUDAStream.h"
#include "cuda/CudaAllocator.h"

TEST(cudaAllocator, run_successfully) {
  simpletorch::globalContext().lazyInitCUDA();
  c10::cuda::cuda_allocator::CUDAAllocator *allocator = c10::cuda::cuda_allocator::get();
  c10::DataPtr ptr = allocator->allocate(64 * 64 * sizeof(int));
}

TEST(cudaAllocator, get_stream_from_pool) {
  simpletorch::globalContext().lazyInitCUDA();
  c10::cuda::CUDAStream cudaStream = c10::cuda::getStreamFromPool(false);
  EXPECT_EQ(static_cast<int>(cudaStream.device_index()), 0);

  cudaStream_t stream_ = cudaStream.stream();
  c10::StreamId stream_id = cudaStream.unwrap().id();
  c10::DeviceIndex device = cudaStream.device_index();
  // compare with hardcoded id
  EXPECT_EQ(stream_, c10::cuda::danger_return_stream(
    0, device,
    static_cast<size_t>(stream_id >> (4 + 1))));
}

TEST(cudaAllocator, get_high_priority_stream_from_pool) {
  simpletorch::globalContext().lazyInitCUDA();
  c10::cuda::CUDAStream cudaStream = c10::cuda::getStreamFromPool(true);
  EXPECT_EQ(static_cast<int>(cudaStream.device_index()), 0);

  cudaStream_t stream_ = cudaStream.stream();
  c10::StreamId stream_id = cudaStream.unwrap().id();
  int max_priority = c10::cuda::max_stream_priority();
  c10::DeviceIndex device = cudaStream.device_index();
  // compare with hardcoded id
  EXPECT_EQ(stream_, c10::cuda::danger_return_stream(
    max_priority - 1, device,
    static_cast<size_t>(stream_id >> (4 + 1))));
}

TEST(cudaAllocator, call_several_times_get_from_pool) {
  simpletorch::globalContext().lazyInitCUDA();
  c10::cuda::CUDAStream cudaStream = c10::cuda::getStreamFromPool(true);
  cudaStream = c10::cuda::getStreamFromPool(true);
  cudaStream = c10::cuda::getStreamFromPool(true);
  EXPECT_EQ(static_cast<int>(cudaStream.device_index()), 0);

  cudaStream_t stream_ = cudaStream.stream();
  c10::StreamId stream_id = cudaStream.unwrap().id();
  int max_priority = c10::cuda::max_stream_priority();
  c10::DeviceIndex device = cudaStream.device_index();
  int idx = static_cast<size_t>(stream_id >> (4 + 1));
  EXPECT_EQ(idx, 2);

  // compare with hardcoded id
  EXPECT_EQ(stream_, c10::cuda::danger_return_stream(
    max_priority - 1, device,
    idx));
}
