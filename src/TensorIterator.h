#pragma once

#include "Device.h"
#include "DeviceType.h"
#include "ScalarType.h"
#include "Tensor.h"
#include "utils/DimVector.h"
#include <cstdint>
#include <utility>
#include <vector>
namespace simpletorch {

class TensorIteratorConfig;

struct OperandInfo {
friend struct TensorIterator;
OperandInfo() = default;
OperandInfo(const OperandInfo&) = default;
OperandInfo& operator=(const OperandInfo&) = default;
OperandInfo(OperandInfo&&) noexcept = default;
OperandInfo& operator=(OperandInfo&&) noexcept = default;

explicit OperandInfo(Tensor&& t): target_dtype(t.scalar_type()),
  current_dtype(t.scalar_type()), tensor_(std::move(t)) { }

~OperandInfo() = default;

private:
  /// The desired device and type for the operand. For inputs, this specifies
  /// that the input should be converted to this type if necessary. For outputs,
  /// this specifies which type to allocate. target_dtype and device are
  /// initialized with the dtype and device of the tensor but during type
  /// promotion target_dtype value can become different from tensor's dtype
  /// also, during type promotion target_dtype and device can be set for an
  /// undefined tensor so that tensor can be properly constructed later.
  c10::ScalarType target_dtype = c10::ScalarType::Undefined;
    // Caches dtype of the tensor, because scalar_type is an expensive operation
  // If dtype of the tensor is changed (e.g. as a result of type promotion or in
  // allocate_outputs), this
  // value should be changed too.
  c10::ScalarType current_dtype = c10::ScalarType::Undefined;
  Tensor tensor_;

  /// The data pointer. This may be different from tensor->data_ptr() if the
  /// iterator is split.
  void* data = nullptr;
};

struct TensorIterator {
public:
  friend struct TensorIteratorConfig;
  c10::ScalarType dtype(int64_t arg = 0) const {
    return operands_[arg].current_dtype;
  }

  int ntensors() const {
    return static_cast<int>(operands_.size());
  }

  c10::Device device(int64_t arg = 0) const {
    // hardcode to cuda device for now
    return c10::Device(c10::DeviceType::CUDA);
  }
  int64_t numel() const {
    int64_t numel = 1;
    for (auto size : shape_) {
      numel *= size;
    }
    return numel;
  }

  bool can_use_32bit_indexing() const;
  bool is_contiguous() const {
    return true;
  }
  int ninputs() const {
    return ntensors() - noutputs();
  }

  int noutputs() const {
    return num_outputs_;
  }
  void* data_ptr(int64_t arg) const {
    return operands_[arg].data;
  }
private:
  void build(const TensorIteratorConfig& config);
  void populate_operands(const TensorIteratorConfig& config);
  void compute_shape(const TensorIteratorConfig&);
  int num_outputs_;
  std::vector<OperandInfo> operands_;

  /// Whether or not all operands have the same shape and are 1d+. Having all
  /// the same shape affects whether or not the iterator is eligible for fast
  /// setup.
  bool all_ops_same_shape_ = false;
  c10::DimVector shape_;
};

class TensorIteratorConfig final {
public:
  friend struct TensorIterator;
  TensorIteratorConfig& add_output(const Tensor& output) {
    return add_borrowed_output(output);
  }
  TensorIteratorConfig& add_borrowed_output(const Tensor& output);
  TensorIterator build() {
    TensorIterator iter;
    iter.build(*this);
    return iter;
  }
private:
  // we use Tensor instead of the MayOwned<Tensor> in the original implementation
  std::vector<Tensor> tensor_;
  int num_outputs_ = 0;
  int num_inputs_ = 0;
};

};
