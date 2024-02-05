#pragma once

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

class Allocator {
public:
    virtual UniqueDataPtr allocate(int64_t n) const = 0;
    virtual ~Allocator() = default;
};
}