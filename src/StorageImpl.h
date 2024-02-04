#pragma once

#include "Allocator.h"
#include <cstdint>
#include <utility>

namespace simpletorch {
class StorageImpl {
public:
    StorageImpl(const StorageImpl& rhs)=delete;
    StorageImpl& operator=(const StorageImpl& rhs)=delete;
    StorageImpl(StorageImpl&& rhs) noexcept=delete;
    StorageImpl& operator=(StorageImpl&& rhs) noexcept=delete;
    void* unsafe_get_data() {
        return data_ptr_.get_data();
    }

    StorageImpl(
        int64_t size_bytes,
        c10::UniqueDataPtr&& data_ptr,
        c10::Allocator* allocator)
        : size_bytes_(size_bytes), data_ptr_(std::move(data_ptr)), allocator_(allocator) {};
    
    StorageImpl(int64_t size_bytes, c10::Allocator* allocator)
        :size_bytes_(size_bytes), data_ptr_(std::move(allocator->allocate(size_bytes))), allocator_(allocator) {}
    ~StorageImpl()=default;
private:
    c10::UniqueDataPtr data_ptr_;
    int64_t size_bytes_;
    c10::Allocator* allocator_;
};

}