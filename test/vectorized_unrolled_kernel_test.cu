#include <gtest/gtest.h>
#include "cuda/CUDAException.h"
#include "cuda/CUDAStream.h"
#include "cuda/Loops.cuh"
#include "utils/Array.h"
#include <cuda_runtime.h>

struct FillFunctor {
  FillFunctor(int v): value(v) {}

  __device__ __forceinline__ int operator() () const {
    return value;
  }
private:
  int value;
};

struct AddFunctor {
  __device__ __forceinline__ int operator() (int a, int b) const {
    return a + b;
  }
};

TEST(KernelTest, vectorized_kernel_test_fill) {
  int n = 100000;
  int *d_a;
  int *h_a = new int[n];

  // this is actually dangerous, because it's possible the stream is switched
  C10_CUDA_CHECK(cudaMallocAsync(&d_a, n * sizeof(int), c10::cuda::getCurrentCUDAStream()));

  auto data = c10::Array<decltype(d_a), 1>(d_a);
  c10::cuda::launch_vectorized_kernel(n, FillFunctor(100), data);

  C10_CUDA_CHECK(cudaMemcpyAsync(h_a, d_a, n * sizeof(int), cudaMemcpyDeviceToHost,
    c10::cuda::getCurrentCUDAStream()));
  cudaDeviceSynchronize();

  for (int i = 0; i < n; i++) {  
    EXPECT_EQ(h_a[i], 100);
  }  

  cudaFree(d_a);  
  delete[] h_a;
}

TEST(KernelTest, unroll_kernel_test_fill) {
  int n = 100000;
  int *d_a;
  int *h_a = new int[n];

  // this is actually dangerous, because it's possible the stream is switched
  C10_CUDA_CHECK(cudaMallocAsync(&d_a, n * sizeof(int), c10::cuda::getCurrentCUDAStream()));

  auto data = c10::Array<decltype(d_a), 1>(d_a);
  c10::cuda::launch_unrolled_kernel(n, FillFunctor(100), data);

  C10_CUDA_CHECK(cudaMemcpyAsync(h_a, d_a, n * sizeof(int), cudaMemcpyDeviceToHost,
    c10::cuda::getCurrentCUDAStream()));
  cudaDeviceSynchronize();

  for (int i = 0; i < n; i++) {  
    EXPECT_EQ(h_a[i], 100);
  }  

  cudaFree(d_a);  
  delete[] h_a;
}

TEST(KernelTest, vectorized_kernel_test_add) {
  int n = 100000;
  int *d_a, *d_b, *d_c;
  int *h_sum = new int[n];

  // this is actually dangerous, because it's possible the stream is switched
  C10_CUDA_CHECK(cudaMallocAsync(&d_a, n * sizeof(int), c10::cuda::getCurrentCUDAStream()));
  C10_CUDA_CHECK(cudaMallocAsync(&d_b, n * sizeof(int), c10::cuda::getCurrentCUDAStream()));
  C10_CUDA_CHECK(cudaMallocAsync(&d_c, n * sizeof(int), c10::cuda::getCurrentCUDAStream()));

  auto data = c10::Array<decltype(d_a), 3>();
  data[0] = d_c;
  data[1] = d_b;
  data[2] = d_a;

  c10::cuda::launch_vectorized_kernel(n, FillFunctor(100), c10::Array<decltype(d_a), 1>(d_a));
  c10::cuda::launch_vectorized_kernel(n, FillFunctor(200), c10::Array<decltype(d_a), 1>(d_b));
  c10::cuda::launch_vectorized_kernel(n, AddFunctor(), data);

  C10_CUDA_CHECK(cudaMemcpyAsync(h_sum, d_c, n * sizeof(int), cudaMemcpyDeviceToHost,
    c10::cuda::getCurrentCUDAStream()));
  cudaDeviceSynchronize();

  for (int i = 0; i < n; i++) {  
    EXPECT_EQ(h_sum[i], 300);
  }  

  cudaFree(d_a);  
  cudaFree(d_b);  
  cudaFree(d_c);  
  delete[] h_sum;
}

TEST(KernelTest, unroll_kernel_test_add) {
  int n = 100000;
  int *d_a, *d_b, *d_c;
  int *h_sum = new int[n];

  // this is actually dangerous, because it's possible the stream is switched
  C10_CUDA_CHECK(cudaMallocAsync(&d_a, n * sizeof(int), c10::cuda::getCurrentCUDAStream()));
  C10_CUDA_CHECK(cudaMallocAsync(&d_b, n * sizeof(int), c10::cuda::getCurrentCUDAStream()));
  C10_CUDA_CHECK(cudaMallocAsync(&d_c, n * sizeof(int), c10::cuda::getCurrentCUDAStream()));

  auto data = c10::Array<decltype(d_a), 3>();
  data[0] = d_c;
  data[1] = d_b;
  data[2] = d_a;

  c10::cuda::launch_unrolled_kernel(n, FillFunctor(100), c10::Array<decltype(d_a), 1>(d_a));
  c10::cuda::launch_unrolled_kernel(n, FillFunctor(200), c10::Array<decltype(d_a), 1>(d_b));
  c10::cuda::launch_unrolled_kernel(n, AddFunctor(), data);

  C10_CUDA_CHECK(cudaMemcpyAsync(h_sum, d_c, n * sizeof(int), cudaMemcpyDeviceToHost,
    c10::cuda::getCurrentCUDAStream()));
  cudaDeviceSynchronize();

  for (int i = 0; i < n; i++) {  
    EXPECT_EQ(h_sum[i], 300);
  }  

  cudaFree(d_a);  
  cudaFree(d_b);  
  cudaFree(d_c);  
  delete[] h_sum;
}
