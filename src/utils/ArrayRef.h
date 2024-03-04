#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace c10 {

/// ArrayRef - Represent a constant reference to an array (0 or more elements
/// consecutively in memory), i.e. a start pointer and a length.  It allows
/// various APIs to take consecutive elements easily and conveniently.
///
/// This class does not own the underlying data, it is expected to be used in
/// situations where the data resides in some other buffer, whose lifetime
/// extends past that of the ArrayRef. For this reason, it is not in general
/// safe to store an ArrayRef.
///
/// This is intended to be trivially copyable, so it should be passed by
/// value.
template<typename T>
class ArrayRef final {
public:
using iterator = const T*;
using const_iterator = const T*;
using size_type = size_t;
using value_type = T;

constexpr ArrayRef(const T& oneElement): data_(&oneElement), length_(1) {}
constexpr ArrayRef(const T* data, size_type length): data_(data), length_(length) {}
constexpr ArrayRef(const T*begin, const T*end): data_(begin), length_(end - begin) {}

template <typename A>
ArrayRef(const std::vector<T, A>& vec): data_(vec.data()), length_(vec.size()) {}

template <size_t N>
constexpr ArrayRef(std::array<T, N>& arr): data_(arr.data()), length_(N) {}

constexpr iterator begin() const {
  return data_;
}

constexpr iterator end() const {
  return data_ + length_;
}

  constexpr bool empty() const {
    return length_ == 0;
  }

  constexpr const T* data() const {
    return data_;
  }

  /// size - Get the array size.
  constexpr size_t size() const {
    return length_;
  }

private:
  // the start of the array, in an external buffer
  const T* data_;
  size_type length_;
};

using Int64ArrayRef = ArrayRef<int64_t>;
}
