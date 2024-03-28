
#include "Scalar.h"
#include "ScalarType.h"
#include "Tensor.h"
#include "cuda/CUDAException.h"
#include "cuda/CUDAStream.h"
#include "ops/FillOps.h"
#include <glog/logging.h>

template<typename T>
void test_fill(c10::ScalarType type) {
  std::array<int64_t, 2> size_array {500, 600};
	c10::Int64ArrayRef size(size_array);
  simpletorch::Tensor tensor = simpletorch::ops::tensor_fill::call(
    size, type, c10::Scalar(100));
  T* h_a = new T[500 * 600];

  C10_CUDA_CHECK(cudaMemcpyAsync(h_a, tensor.const_data_ptr(), 500 * 600 * sizeof(T), cudaMemcpyDeviceToHost,
    c10::cuda::getCurrentCUDAStream()));
  cudaDeviceSynchronize();
  
  for (int i = 0; i < 500 * 600; i++) {  
    TORCH_CHECK(h_a[i] == 100, "h_a[", i, "]=", h_a[i]);
  }  
  delete[] h_a;
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  test_fill<int>(c10::ScalarType::Int);
  test_fill<float>(c10::ScalarType::Float);
}