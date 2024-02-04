#pragma once

#include "Allocator.h"

namespace c10 {

class NaiveCpuAllocator : public Allocator {
public:
    UniqueDataPtr allocate(int64_t n) const override;
};
}
