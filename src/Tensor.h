#pragma once

#include "TensorImpl.h"
#include <memory>

namespace simpletorch {
class Tensor {
public:
	Tensor() = default;
	explicit Tensor(std::shared_ptr<TensorImpl> &&tensor_impl): tensor_impl_(std::move(tensor_impl)) {}
	void* get_unsafe_data() {
		return tensor_impl_->unsafe_get_data();
	}

	TensorImpl* unsafeGetTensorImpl() const {
		return tensor_impl_.get();
	}

	c10::Int64ArrayRef get_sizes() const {
		return tensor_impl_->get_sizes();	
	}

	~Tensor()=default;
protected:
  std::shared_ptr<TensorImpl> tensor_impl_;
};
}
