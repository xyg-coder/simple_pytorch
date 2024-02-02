#include "Allocator.h"
# include "CpuAllocator.h"

int main() {
    c10::NaiveCpuAllocator allocator;
    c10::UniqueDataPtr dataPtr = allocator.allocate(100 * sizeof(int));
}
