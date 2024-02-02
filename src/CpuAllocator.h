#pragma once

#include "Allocator.h"

namespace c10 {

class NaiveCpuAllocator : public Allocator {
public:
    UniqueDataPtr allocate(size_t n) const;
    ~NaiveCpuAllocator() {};
};
}
