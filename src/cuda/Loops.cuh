#pragma once

#include "cuda/OffsetCalculator.cuh"
#include "cuda/ThreadConstants.h"
#include "macros/Macros.h"
#include "cuda/MemoryAccess.cuh"
#include "utils/Metaprogramming.h"
#include <tuple>
#include <utility>

namespace c10::cuda {

template <typename func_t, typename policy_t>
__device__ inline void elementwise_kernel_helper(func_t f, policy_t policy) {
  using traits = c10::guts::function_traits<func_t>;
  using return_t = typename traits::return_type;
  using args_t = typename traits::parameter_types;

  int idx = blockIdx.x;
  return_t result[thread_work_size()];
  args_t args[thread_work_size()];

  // load
  policy.load(args, idx);

  // compute
  #pragma unroll
  for (int i = 0; i < thread_work_size(); ++i) {
    if (policy.check_inbounds(i)) {
      result[i] = std::apply(
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
  using traits = guts::function_traits<func_t>;
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
  }
}

template<typename func_t, typename array_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_element_wise_kernel(int N, array_t data) {
  using traits = guts::function_traits<func_t>;
  auto input_calc = TrivialOffsetCalculator<traits::number_of_parameters>();
  auto output_calc = TrivialOffsetCalculator<1>();
  auto loader = policies::LoadWithoutCast();
  auto storer = policies::StoreWithoutCast();
  auto policy = policies::unroll<
    array_t, decltype(input_calc), decltype(output_calc), policies::LoadWithoutCast,
    policies::StoreWithoutCast>(
      data, N, input_calc, output_calc, loader, storer);
}

} // namespace c10::cuda
