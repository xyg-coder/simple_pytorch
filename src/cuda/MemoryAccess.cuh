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

} // anonymous namespace

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

} // namespace c10::cuda
