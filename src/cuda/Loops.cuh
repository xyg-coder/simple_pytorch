#pragma once

#include "cuda/CUDAException.h"
#include "cuda/CUDAStream.h"
#include "cuda/OffsetCalculator.cuh"
#include "cuda/ThreadConstants.h"
#include "macros/Macros.h"
#include "cuda/MemoryAccess.cuh"
#include "utils/Apply.h"
#include "utils/Exception.h"
#include "utils/Logging.h"
#include "utils/Metaprogramming.h"
#include <cstdint>
#include <iostream>
#include <limits>
#include <tuple>
#include <utility>

namespace c10::cuda {

template <typename func_t, typename policy_t>
C10_HOST_DEVICE inline void elementwise_kernel_helper(func_t f, policy_t policy) {
  using traits = c10::guts::infer_function_traits_t<func_t>;
  using return_t = typename traits::return_type;
  using args_t = typename traits::ArgsTuple;

  int idx = blockIdx.x;
  return_t result[thread_work_size()];
  args_t args[thread_work_size()];

  // load
  policy.load(args, idx);

  // compute
  #pragma unroll
  for (int i = 0; i < thread_work_size(); ++i) {
    if (policy.check_inbounds(i)) {
      result[i] = c10::guts::apply(
        std::move(f),
        std::move(args[i]));
    }
  }

  // store
  policy.store(result, idx);
}

template<int vec_size, typename func_t, typename array_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void vectorized_elementwise_kernel(int N, func_t f, array_t data) {
  using traits = guts::infer_function_traits_t<func_t>;
  int remaining = N - block_work_size() * blockIdx.x;
  if (remaining >= block_work_size()) {
    elementwise_kernel_helper(
      f, policies::vectorized<vec_size, array_t>(data));
  } else {
    auto input_calc = TrivialOffsetCalculator<traits::number_of_parameters>();
    auto output_calc = TrivialOffsetCalculator<1>();
    auto loader = policies::LoadWithoutCast();
    auto storer = policies::StoreWithoutCast();
    auto policy = policies::unroll<
      array_t, decltype(input_calc), decltype(output_calc), policies::LoadWithoutCast,
      policies::StoreWithoutCast>(
        data, remaining, input_calc, output_calc, loader, storer
      );
    elementwise_kernel_helper(f, policy);
  }
}

template<
  typename func_t,
  typename array_t,
  typename input_calc_t,
  typename output_calc_t,
  typename loader_t,
  typename storer_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_element_wise_kernel(
  int N, func_t f, array_t data, input_calc_t ic, output_calc_t oc, loader_t l, storer_t s) {
  using traits = guts::infer_function_traits_t<func_t>;
  auto policy = policies::unroll<
    array_t, decltype(ic), decltype(oc), policies::LoadWithoutCast,
    policies::StoreWithoutCast>(
      data, N, ic, oc, l, s);
  elementwise_kernel_helper(f, policy);
}

template<typename func_t, typename array_t>
static inline void launch_vectorized_kernel(
  int64_t N, const func_t& f, array_t data) {
  
  TORCH_CHECK(N > 0 && N <= std::numeric_limits<int32_t>::max());
  using traits = guts::infer_function_traits_t<func_t>;
  int64_t grid = (N + block_work_size() - 1) / block_work_size();
  auto stream = getCurrentCUDAStream();
  int vec_size = memory::can_vectorize_up_to<func_t>(data);
  LOG_INFO("launch_vectorized_kernel, vec_size=", vec_size);
  switch (vec_size) {
    case 4:
      vectorized_elementwise_kernel<4><<<grid, num_threads(), 0, stream>>>(N, f, data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 2:
      vectorized_elementwise_kernel<2><<<grid, num_threads(), 0, stream>>>(N, f, data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 1: {
      auto input_calc = TrivialOffsetCalculator<traits::number_of_parameters>();
      auto output_calc = TrivialOffsetCalculator<1>();
      auto loader = policies::LoadWithoutCast();
      auto storer = policies::StoreWithoutCast();
      unrolled_element_wise_kernel<<<
        grid, num_threads(), 0, stream>>>(N, f, data, input_calc, output_calc, loader, storer);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    }
    default:
      TORCH_CHECK(false, "Unexpected vectorization size");
  }
}

template<typename func_t, typename array_t>
static inline void launch_unrolled_kernel(
  int64_t N, const func_t& f, array_t data) {

  TORCH_CHECK(N > 0 && N <= std::numeric_limits<int32_t>::max());
  using traits = guts::infer_function_traits_t<func_t>;
  int64_t grid = (N + block_work_size() - 1) / block_work_size();
  auto stream = getCurrentCUDAStream();

  auto input_calc = TrivialOffsetCalculator<traits::number_of_parameters>();
  auto output_calc = TrivialOffsetCalculator<1>();
  auto loader = policies::LoadWithoutCast();
  auto storer = policies::StoreWithoutCast();
  unrolled_element_wise_kernel<<<
    grid, num_threads(), 0, stream>>>(N, f, data, input_calc, output_calc, loader, storer);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace c10::cuda
