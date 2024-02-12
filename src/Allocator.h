#pragma once

#include "Device.h"
#include "DeviceType.h"
#include <functional>
#include <memory>
namespace c10 {

// use std::function instead of function pointer: this provides ability to add hooks (see allocator_test.cpp for one example)
using DeleteFnPtr = std::function<void(void*)>;

void deleteNothing(void*);

// A DataPtr is a unique pointer (with an attached deleter and some
// context for the deleter) to some memory, which also records what
// device is for its data.
class UniqueDataPtr final{
public:
    UniqueDataPtr(): data_(nullptr), ctx_(nullptr, &deleteNothing) {};
    UniqueDataPtr(void* data): data_(data), ctx_(nullptr, &deleteNothing) {};
    UniqueDataPtr(void* data, void* ctx, DeleteFnPtr deleteFn): data_(data), ctx_(ctx, deleteFn) {};
    void* get_data() {
        return data_;
    }

private:
    void* data_;
    // the deleter is guaranteed to be called when the unique pointer is destructed and the context is not null
    std::unique_ptr<void, DeleteFnPtr> ctx_;
};

class DataPtr {
public:
    DataPtr(): ptr_(), device_(DeviceType::CPU) {};
    DataPtr(void* data, DeviceType device): ptr_(data), device_(device) {};
    DataPtr(void* data, void* ctx, DeleteFnPtr deleteFn, DeviceType device): ptr_(data, ctx, deleteFn), device_(device) {};
    void *get_data() {
        return ptr_.get_data();
    }
private:    
    UniqueDataPtr ptr_;
    Device device_;
};


class Allocator {
public:
    virtual DataPtr allocate(int64_t n) const = 0;
    virtual ~Allocator() = default;
};
}
