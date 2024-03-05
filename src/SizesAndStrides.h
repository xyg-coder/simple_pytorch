#pragma once

#include "utils/ArrayRef.h"
#include "utils/Exception.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#define SIZES_AND_STRIDES_MAX_SIZE 5

namespace c10 {
class SizesAndStrides final {
public:
	using sizes_iterator = int64_t*;
	using sizes_const_iterator = const int64_t*;
	using strides_iterator = int64_t*;
	using strides_const_iterator = const int64_t*;

	SizesAndStrides() {
		size_ = 1;
		storage_[0] = 0;
		storage_[1] = 1;
	}
	~SizesAndStrides()=default;
	SizesAndStrides(c10::Int64ArrayRef sizes, c10::Int64ArrayRef strides) {
		throw std::logic_error("SizesAndStrides(c10::Int64ArrayRef sizes, c10::Int64ArrayRef strides) is not implemented yet");
	}
	SizesAndStrides(const SizesAndStrides &rhs): size_(rhs.size_) {
		copy_data_inline(rhs);
	}

	SizesAndStrides& operator=(const SizesAndStrides& rhs) {
		if (this == &rhs) {
			return *this;
		}
		copy_data_inline(rhs);
		size_ = rhs.size_;
		return *this;
	}
	SizesAndStrides(SizesAndStrides&& rhs) noexcept: size_(rhs.size_) {
		copy_data_inline(rhs);
		rhs.size_ = 0;
	}
	SizesAndStrides& operator=(SizesAndStrides&& rhs) noexcept {
		if (this == &rhs) {
			return *this;
		}
		copy_data_inline(rhs);
		size_ = rhs.size_;
		return *this;
	}

	void set_sizes(Int64ArrayRef newSizes) {
		resize(newSizes.size());
		std::copy(newSizes.begin(), newSizes.end(), &storage_[0]);
	}

	Int64ArrayRef sizes_arrayref() const noexcept {
		return Int64ArrayRef{sizes_data(), size()};
	}

private:
	void copy_data_inline(const SizesAndStrides& rhs) {
		memcpy(storage_, rhs.storage_, sizeof(storage_));
	}

	size_t size() const noexcept {
		return size_;
	}

	void resize(size_t newSize) {
		const auto oldSize = size();
		if (newSize == oldSize) {
			return;
		}
		TORCH_CHECK(newSize <= SIZES_AND_STRIDES_MAX_SIZE, "Tensor size of more than ",
			SIZES_AND_STRIDES_MAX_SIZE, " is not supported.")

		if (oldSize < newSize) {
			const auto bytesToZero = (newSize - oldSize) * sizeof(storage_[0]);
			memset(&storage_[oldSize], 0, bytesToZero);
			memset(&storage_[SIZES_AND_STRIDES_MAX_SIZE + oldSize], 0, bytesToZero);
		}
		size_ = newSize;
	}

	const int64_t* sizes_data() const noexcept {
		return &storage_[0];
	}

	int64_t* strides_data() noexcept {
		return &storage_[SIZES_AND_STRIDES_MAX_SIZE];
	}

	size_t size_;
	// first size_ elements store the sizes and second size_ elements store the strides
	int64_t storage_[SIZES_AND_STRIDES_MAX_SIZE * 2]{};
};
}
