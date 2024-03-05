#pragma once

#include "SizesAndStrides.h"
#include "Storage.h"
#include "utils/ArrayRef.h"
#include <utility>

namespace simpletorch {
class TensorImpl {
public:
	// TensorImpl should have one single source, disable copy here
	TensorImpl(const TensorImpl& rhs)=delete;
	TensorImpl& operator=(const TensorImpl& rhs)=delete;
	TensorImpl(TensorImpl&& rhs) noexcept=delete;
	TensorImpl& operator=(TensorImpl&& rhs) noexcept=delete;

	TensorImpl(Storage &&storage): storage(std::move(storage)) {};
	TensorImpl(Storage &&storage, c10::Int64ArrayRef sizes, c10::Int64ArrayRef strides)
		: storage(std::move(storage)), size_and_strides_(sizes, strides) {};
	void* unsafe_get_data() {
		return storage.unsafe_get_data();
	}

	void set_sizes_contiguous(c10::Int64ArrayRef new_sizes) {
		size_and_strides_.set_sizes(new_sizes);
	}

	c10::Int64ArrayRef get_sizes() const {
		return size_and_strides_.sizes_arrayref();
	}

	~TensorImpl()=default;
protected:
	Storage storage;
	c10::SizesAndStrides size_and_strides_;
};
}