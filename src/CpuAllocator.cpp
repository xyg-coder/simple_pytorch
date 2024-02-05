#include "CpuAllocator.h"
#include "Allocator.h"
#include <cassert>
#include <cstdint>
#include <iostream>

void c10::deleteNaiveCpuData(void *data) {
    if (data != nullptr) {
        int *typedArrayPtr = static_cast<int*>(data);
        delete[] typedArrayPtr;
        std::cout << "deleteNaiveCpuData is called" << std::endl;
    }
}

c10::UniqueDataPtr c10::NaiveCpuAllocator::allocate(int64_t n) const {
    // currently only suppport int size
    assert(n % sizeof(int) == 0);
    int* arr = new int[n / sizeof(int)];
    c10::UniqueDataPtr result(
        static_cast<void*>(arr),
        static_cast<void*>(arr),
        delete_fn_);
    return result;
}
