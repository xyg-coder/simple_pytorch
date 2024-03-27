#pragma once

#include "StorageImpl.h"
#include <memory>

namespace simpletorch {

class Storage {
public:
Storage() = default;
Storage(std::shared_ptr<StorageImpl>&& storage_impl): storage_impl_(std::move(storage_impl)) {}
const void *data() const {
    return storage_impl_->unsafe_get_data();
}

protected:
    std::shared_ptr<StorageImpl> storage_impl_;
};
}