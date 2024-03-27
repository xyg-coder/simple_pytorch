#pragma once

#include "Device.h"
#include "ScalarType.h"
#include <cstdint>
namespace simpletorch {
struct TensorIterator {
public:
  c10::ScalarType dtype(int64_t arg = 0) const;
  int ntensors() const;
  c10::Device device(int64_t arg = 0) const;
  int64_t numel() const;
  bool can_use_32bit_indexing() const;
  bool is_contiguous() const;
  int ninputs() const;
  int noutputs() const;
  void* data_ptr(int64_t arg) const;
private:

};

class TensorIteratorConfig final {
public:
  TensorIterator build();
};

};
