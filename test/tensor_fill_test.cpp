#include "ScalarType.h"
#include "Tensor.h"
#include "cuda/CUDAException.h"
#include "cuda/CUDAStream.h"
#include "ops/FillOps.h"
#include "utils/ArrayRef.h"
#include <array>
#include <cstdint>
#include <gtest/gtest.h>

TEST(TensorFillTest, intFill) {
  std::array<int64_t, 2> size_array {500, 600};
	c10::Int64ArrayRef size(size_array);
  // 100.1 should convert to 100 in int
  simpletorch::Tensor tensor = simpletorch::ops::tensor_fill::call(
    size, c10::ScalarType::Int, c10::Scalar(100.1));
  int* h_a = new int[500 * 600];

  C10_CUDA_CHECK(cudaMemcpyAsync(h_a, tensor.const_data_ptr(), 500 * 600 * sizeof(int),
   cudaMemcpyDeviceToHost,
    c10::cuda::getCurrentCUDAStream()));
  cudaDeviceSynchronize();
  
  for (int i = 0; i < 500 * 600; i++) {  
    EXPECT_EQ(h_a[i], 100);
  }  
  delete[] h_a;
}

TEST(TensorFillTest, floatFill) {
  std::array<int64_t, 2> size_array {500, 600};
	c10::Int64ArrayRef size(size_array);
  // 100.1 should convert to 100 in int
  simpletorch::Tensor tensor = simpletorch::ops::tensor_fill::call(
    size, c10::ScalarType::Float, c10::Scalar(100.1));
  float* h_a = new float[500 * 600];

  C10_CUDA_CHECK(cudaMemcpyAsync(h_a, tensor.const_data_ptr(), 500 * 600 * sizeof(float),
   cudaMemcpyDeviceToHost,
    c10::cuda::getCurrentCUDAStream()));
  cudaDeviceSynchronize();
  
  for (int i = 0; i < 500 * 600; i++) {  
    EXPECT_EQ(static_cast<int>(h_a[i] * 10), 1001);
  }  
  delete[] h_a;
}
