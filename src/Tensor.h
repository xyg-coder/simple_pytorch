#pragma once

#include "ScalarType.h"
#include "TensorImpl.h"
#include <memory>

namespace simpletorch {
class Tensor {
public:
	Tensor() = default;
	explicit Tensor(std::shared_ptr<TensorImpl> &&tensor_impl): tensor_impl_(std::move(tensor_impl)) {}

	TensorImpl* unsafeGetTensorImpl() const {
		return tensor_impl_.get();
	}

	c10::Int64ArrayRef get_sizes() const {
		return tensor_impl_->get_sizes();	
	}

	c10::ScalarType scalar_type() const {
		return tensor_impl_->dtype().toScalarType();
	}

	const void* const_data_ptr() const {
		return tensor_impl_->data();
	}

	~Tensor()=default;
protected:
  std::shared_ptr<TensorImpl> tensor_impl_;
};
} // namespace simpletorch
