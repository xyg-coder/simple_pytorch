#include "CpuAllocator.h"
#include "Allocator.h"
#include "utils/Logging.h"
#include <cassert>
#include <cstdint>
#include <glog/logging.h>

void c10::deleteNaiveCpuData(void *data) {
    if (data != nullptr) {
        int *typedArrayPtr = static_cast<int*>(data);
        delete[] typedArrayPtr;
        LOG_INFO("deleteNaiveCpuData is called");
    }
}

c10::DataPtr c10::NaiveCpuAllocator::allocate(int64_t n) const {
    // currently only suppport int size
    assert(n % sizeof(int) == 0);
    int* arr = new int[n / sizeof(int)];
    c10::DataPtr result(
        static_cast<void*>(arr),
        static_cast<void*>(arr),
        delete_fn_,
        DeviceType::CPU);
    return result;
}
