#include "TensorIterator.h"
#include "Tensor.h"
#include "utils/DimVector.h"
#include "utils/Exception.h"
#include <cstdint>
#include <limits>

namespace simpletorch {

void TensorIterator::build(const TensorIteratorConfig& config) {
  populate_operands(config);
  compute_shape(config);

  for (auto& op : operands_) {
    op.data = const_cast<void*>(op.tensor_.const_data_ptr());
  }
}

void TensorIterator::compute_shape(const TensorIteratorConfig& config) {
  all_ops_same_shape_ = true;
  bool has_scalars = false;
  bool has_tensors = false;
  for (auto& op: operands_) {
    auto shape = op.tensor_.get_sizes();
    if (shape.empty()) {
      has_scalars = true;
    } else {
      has_tensors = true;
    }

    if (has_scalars && has_tensors) {
      all_ops_same_shape_ = false;
    }

    if (shape_.empty()) {
      shape_ = c10::DimVector(shape.begin(), shape.end());
    }
  }
  TORCH_CHECK(all_ops_same_shape_, "Currently only operands of the same shape are supported");
}

void TensorIterator::populate_operands(const TensorIteratorConfig& config) {
  
  for (int i = 0; i < config.tensor_.size(); ++i) {
    auto& tensor = config.tensor_[i];
    operands_.emplace_back(tensor);
  }
  num_outputs_ = config.num_outputs_;
}

TensorIteratorConfig& TensorIteratorConfig::add_borrowed_output(const Tensor& output) {
  TORCH_CHECK(num_inputs_ == 0, "You should add all outputs before adding any input");
  tensor_.push_back(output);
  num_outputs_++;
  return *this;
}

bool TensorIterator::can_use_32bit_indexing() const {
  int64_t max_value = std::numeric_limits<int32_t>::max();
  return numel() <= max_value;
  // ignore the strides for now
}
} // namespace simpletorch
