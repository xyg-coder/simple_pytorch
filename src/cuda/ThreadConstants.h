#pragma once

#include "macros/Macros.h"
#include <cstdint>

#ifndef GPU_LAMBDA
#define GPU_LAMBDA __host__ __device__
#endif

GPU_LAMBDA constexpr uint32_t num_threads() {
  return C10_WRAP_SIZE * 4;
}

// how many works are performed per thread
GPU_LAMBDA constexpr int thread_work_size() {
  return 4;
}

// how many works are performed per block
GPU_LAMBDA constexpr int block_work_size() {
  return thread_work_size() * num_threads();
}
