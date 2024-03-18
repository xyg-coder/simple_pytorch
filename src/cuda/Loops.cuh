#pragma once

#include "cuda/ThreadConstants.h"
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
} // namespace c10::cuda
