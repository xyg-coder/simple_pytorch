#include "CpuAllocator.h"
#include "Allocator.h"
#include <cassert>
#include <cstdint>
#include <glog/logging.h>

void c10::deleteNaiveCpuData(void *data) {
    if (data != nullptr) {
        int *typedArrayPtr = static_cast<int*>(data);
        delete[] typedArrayPtr;
        LOG(INFO) << "deleteNaiveCpuData is called";
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
