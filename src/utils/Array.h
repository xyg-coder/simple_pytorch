#pragma once

#include "macros/Macros.h"
namespace c10 {
template <typename T, int size_>
struct Array {
  T data[size_];

  C10_HOST_DEVICE T operator[](int i) const {
    return data[i];
  }

  C10_HOST_DEVICE T& operator[](int i) {
    return data[i];
  }

  Array() = default;
  Array(const Array&) = default;
  Array& operator=(const Array&) = default;

  static constexpr int size(){return size_;}
  // Fill the array with x.
  C10_HOST_DEVICE Array(T x) {
    for (int i = 0; i < size_; i++) {
      data[i] = x;
    }
  }
};
}
