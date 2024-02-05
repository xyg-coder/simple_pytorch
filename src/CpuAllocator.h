#pragma once

#include "Allocator.h"

namespace c10 {

void deleteNaiveCpuData(void *data);

class NaiveCpuAllocator : public Allocator {
public:
    NaiveCpuAllocator(): delete_fn_(&deleteNaiveCpuData) {};
    NaiveCpuAllocator(c10::DeleteFnPtr fn_ptr): delete_fn_(fn_ptr) {};
    UniqueDataPtr allocate(int64_t n) const override;
private:
    DeleteFnPtr delete_fn_;
};
}
