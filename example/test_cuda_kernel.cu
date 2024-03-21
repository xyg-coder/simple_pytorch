#include "cuda/Loops.cuh"
#include "utils/Array.h"
#include "utils/Exception.h"
#include <stdio.h>
#include <cuda_runtime.h>
  
struct FillFunctor {
  FillFunctor(int v): value(v) {}

  __device__ __forceinline__ int operator() () const {
    return value;
  }
private:
  int value;
};
  
int main() {  
  int n = 100000;
  int *d_a;
  int *h_a = new int[n];

  cudaMalloc(&d_a, n * sizeof(int));  

  auto data = c10::Array<decltype(d_a), 1>(d_a);
  c10::cuda::launch_vectorized_kernel(n, FillFunctor(100), data);

  cudaMemcpy(h_a, d_a, n * sizeof(int), cudaMemcpyDeviceToHost);  

  for (int i = 0; i < n; i++) {  
    TORCH_CHECK(h_a[i] == 100, "h_a[", i, "]=", h_a[i]);
  }  

  cudaFree(d_a);  
  delete[] h_a;

  return 0;  
}
