#pragma once

#include "SizesAndStrides.h"
#include "Storage.h"
#include "utils/ArrayRef.h"
#include "utils/Typeid.h"
#include <utility>

namespace simpletorch {
class TensorImpl {
public:
	// TensorImpl should have one single source, disable copy here
	TensorImpl(const TensorImpl& rhs)=delete;
	TensorImpl& operator=(const TensorImpl& rhs)=delete;
	TensorImpl(TensorImpl&& rhs) noexcept=delete;
	TensorImpl& operator=(TensorImpl&& rhs) noexcept=delete;

	TensorImpl(Storage &&storage, c10::TypeMeta data_type): storage(std::move(storage)), data_type_(data_type) {};

	TensorImpl(Storage &&storage, c10::TypeMeta data_type, c10::Int64ArrayRef sizes, c10::Int64ArrayRef strides)
		: storage(std::move(storage)), data_type_(data_type), size_and_strides_(sizes, strides) {};
	
	const c10::TypeMeta dtype() const {
		return data_type_;
	}

	void set_sizes_contiguous(c10::Int64ArrayRef new_sizes) {
		size_and_strides_.set_sizes(new_sizes);
	}

	c10::Int64ArrayRef get_sizes() const {
		return size_and_strides_.sizes_arrayref();
	}

	inline const void* data() const {
		return data_impl<const void>([this] {return static_cast<const char*>(storage.data());});
	}

	~TensorImpl()=default;
protected:
	Storage storage;
	c10::SizesAndStrides size_and_strides_;
	c10::TypeMeta data_type_;

	template <typename Void, typename Func>
	Void* data_impl(const Func& get_data) const {
		auto* data = get_data();
		static_assert(sizeof(*data) == 1, "get_data must return a byte-addressed pointer.");

		return data;
	}
};
}