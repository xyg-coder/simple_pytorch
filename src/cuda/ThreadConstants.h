#pragma once

#include "macros/Macros.h"
#include <cstdint>

constexpr uint32_t num_threads() {
  return C10_WRAP_SIZE * 4;
}

// how many works are performed per thread
constexpr int thread_work_size() {
  return 4;
}

// how many works are performed per block
constexpr int block_work_size() {
  return thread_work_size() * num_threads();
}
