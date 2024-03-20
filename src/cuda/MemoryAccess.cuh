#pragma once

#include "cuda/ThreadConstants.h"
#include <cstdint>
#include <tuple>

namespace c10::cuda {

namespace {

// What does the `static_unroll` do?
//
// We want to do something like:
//
//    using args_t = typename traits::ArgsTuple;
//    args_t args;
//    #pragma unroll
//    for (int i = 0; i < traits::arity; i++) {
//      std::get<i>(args) = ....
//    }
//
// but unfortunately the above code does not work because
// the template argument has to be a compile time constant
// so `static_unroll` is created to simulate `#pragma unroll`
// using template metaprogramming.
template<template<int i>typename func, int end, int current=0>
struct static_unroll {
  template<typename... Args>
  static inline void with_args(Args&&... args) {
    func<current>::apply(std::forward<Args>(args)...);
    static_unroll<func, end, current+1>::with_args(args...);
  }
};

template<template<int i>typename func, int end>
struct static_unroll<func, end, end> {
  template<typename... Args>
  static inline void with_args(Args... args) { }
};

template<template<int i>typename func, int vec_size, int end, int current=0> 
struct static_unroll_with_vec_size {
  template<typename... Args>
  static inline void with_args(Args&&... args) {
    func<current, vec_size>::apply(std::forward<Args>(args)...);
    static_unroll_with_vec_size<func, vec_size, end, current+1>::with_args(args...);
  }
};

template<template<int i>typename func, int vec_size, int end>
struct static_unroll_with_vec_size<func, vec_size, end, end> {
  template<typename... Args>
  static inline void with_args(Args... args) { }
};

template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};
} // anonymous namespace

namespace policies {
struct LoadWithoutCast {

  template<typename scalar_t>
  __device__ scalar_t load(char* base_ptr, uint32_t offset, int arg=0) {
    scalar_t* base_ptr_scalar = std::reinterpret_cast<scalar_t*>(base_ptr);
    return *(base_ptr_scalar + offset);
  }
};

struct StoreWithoutCast {

  template<typename scalar_t>
  __device__ void store(scalar_t value, char* base_ptr, uint32_t offset, int arg=0) {
    scalar_t* base_ptr_scalar = std::reinterpret_cast<scalar_t*>(base_ptr);
    *(base_ptr_scalar + offset) = value;
  }
};

template<int arg_index>
struct unroll_load_helper {
  template<typename args_t, typename policy_t, typename offset_t, typename loader_t>
  static __device__ void apply(policy_t &self, args_t *args, offset_t offset, loader_t loader, int j, int num_outputs) {
    // `data` hold the data_ptr for tensors [output, input0, input1, ...]
    // each element in the array holds the parameters for the arg_index
    std::get<arg_index>(args[j]) = loader.template load<args_t>(self.data[arg_index + num_outputs], offset[arg_index], arg_index);
  }
};

template<int arg_index, int vec_size>
struct vectorized_load_helper {
  template<typename args_t, typename policy_t>
  static __device__ void apply(policy_t &self, args_t *args, int loop_i) {
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    using vec_args_t = aligned_vector<arg_t, vec_size>;
    auto ptr = std::reinterpret_cast<arg_t*>(data[1 + arg_index]) + idx * block_work_size() + loop_i * num_threads() * vec_size;
    vec_args_t* ptr_ = std::reinterpret_cast<vec_args_t*>(ptr);
    vec_args_t vec_arg = *ptr_;
    #pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      std::get<arg_index>(args[loop_i * vec_size + i]) = vec_arg.val[i];
    }
  }
};

// assumption: all tensor are contiguous
template <typename data_t, typename inp_calc_t, typename out_calc_t,
  typename loader_t, typename storer_t, int num_outputs = 1>
struct unroll {
  // array of data: [output1, output2... input1, input2...]
  data_t data;
  int remaining;
  inp_calc_t input_offset_calculator;
  out_calc_t output_offset_calculator;
  loader_t loader;
  storer_t storer;

  __device__ unroll(data_t data, int remaining, inp_calc_t ic, out_calc_t oc, loader_t l, storer_t s):
    data(data), remaining(remaining), input_offset_calculator(ic), output_offset_calculator(oc), loader(l), storer(s) {}
  
  __device__ check_inbounds(int thread_work_elem) {
    return threadIdx.x + thread_work_elem * num_threads() < remaining;
  }

  // load the inputs of (threadIdx.x, blockIdx.x=idx) into args
  template<typename args_t>
  __device__ inline void load(args_t* args, int idx) {
    // number of arguments
    constexpr int arity = std::tuple_size_v<args_t>;
    int threadIdx = threadIdx.x;

    #pragma unroll
    for (int i = 0; i < thread_work_size(); ++i) {
      if (threadIdx >= remaining) {
        return;
      }

    /*
    // what we want to do:
    for (int j = 0; j < arity; ++j) {
      std::get<j>(args[i]) = loader(data[j+num_outputs], offset);
    }
    // this cannot be done under #pragma unroll, so use metaprogramming
    */
      int linear_idx = thread_idx + block_work_size() * idx;
      auto offset = input_offset_calculator.get(linear_idx);

      static_unroll<unroll_load_helper, arity>::with_args(*this, args, offset, loader, i, num_outputs);
      threadIdx += num_threads();
    }
  }

  template<typename scalar_t>
  __device__ inline void store(scalar_t *from, int idx) {
    int threadIdx = threadIdx.x;

    #pragma unroll
    for (int i = 0; i < thread_work_size(); ++i) {
      if (threadIdx >= remaining) {
        return;
      }
      int linear_idx = threadIdx + block_work_size() * idx;
      // we only have 1 output
      int offset = output_offset_calculator.get(linear_idx)[0];
      storer.store(from[i], data[0], offset);

      threadIdx += num_threads();
    }
  }
};

// Assumption:
// all tensors are contiguous, that is: stride == sizeof(type) for all tensors
// Note:
// Functions in vectorized policy does not do boundary check. It assumes the whole block
// has its job to do. So the reminders should be handled by the caller manually.
template<int vec_size, typename data_t>
struct vectorized {
  data_t data;
  static_assert(thread_work_size() % vec_size == 0, "The workload per thread must be a multiple of vec_size");
  static constexpr int loop_size = thread_work_size() / vec_size;

  __device__ vectorized(data_t data): data(data) {}

  // we assume the whole block is in the bounds for this policy
  __device__ inline constexpr bool check_inbounds(int thread_work_elem) {
    return true;
  }

  template<typename args_t>
  __device__ inline void load(args_t* args, int idx) {
    constexpr narity = std::tuple_size_v<args_t>;
    #pragma unroll
    for (int i = 0; i < loop_size; ++i) {
      static_unroll_with_vec_size<vectorized_load_helper, vec_size, narity>(*this, args, i);
    }
  }

  template<typename scalar_t>
  __device__ inline void store(scalar_t* from, int idx) {
    using vec_scalar = aligned_vector<scalar_t, vec_size>;
    int threadIdx = threadIdx.x;
    int offset = idx * block_work_size() + threadIdx * vec_size;
    scalar_t* to = std::reinterpret_cast<scalar_t*>(data[0]) + offset;
    vec_scalar* to_ = std::reinterpret_cast<vec_scalar*>(to);
    #pragma unroll
    for (int i = 0; i < loop_size; ++i) {
      vec_scalar tmp;
      #pragma unroll
      for (int j = 0; j < vec_size; ++j) {
        tmp.val[j] = from[i * vec_size + j];
      }
      to_[num_threads() * i + threadIdx] = tmp;
    }
  }
};

} // namespace c10::cuda::policies
} // namespace c10::cuda
