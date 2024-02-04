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

    ~Tensor()=default;
protected:
    std::shared_ptr<TensorImpl> tensor_impl_;
};
}
