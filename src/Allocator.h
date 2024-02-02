#pragma once

#include <cstddef>
#include <memory>
namespace c10 {

using DeleteFnPtr = void (*)(void *);

void deleteNothing(void*);

// A DataPtr is a unique pointer (with an attached deleter and some
// context for the deleter) to some memory, which also records what
// device is for its data.
class UniqueDataPtr final{
public:
    UniqueDataPtr(void* data): data_(data), ctx_(nullptr, &deleteNothing) {};
    UniqueDataPtr(void* data, void* ctx, DeleteFnPtr deleteFn): data_(data), ctx_(ctx, deleteFn) {};
private:
    void* data_;
    // the deleter is guaranteed to be called when the unique pointer is destructed and the context is not null
    std::unique_ptr<void, DeleteFnPtr> ctx_;

};

class Allocator {
public:
    virtual UniqueDataPtr allocate(size_t n) const = 0;
    virtual ~Allocator() = default;
};
}