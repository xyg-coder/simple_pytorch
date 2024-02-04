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

    ~TensorImpl()=default;
protected:
    Storage storage;
    c10::SizesAndStrides size_and_strides_;
};
}