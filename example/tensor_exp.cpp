#include "CpuAllocator.h"
#include "Storage.h"
#include "StorageImpl.h"
#include "Tensor.h"
#include "TensorImpl.h"
#include <memory>

int main() {
    c10::NaiveCpuAllocator allocator;

    simpletorch::Storage storage(
        std::make_shared<simpletorch::StorageImpl>(100 * sizeof(int), &allocator));
    simpletorch::Tensor tensor(std::make_shared<simpletorch::TensorImpl>(std::move(storage)));
    simpletorch::Tensor tensor2 = tensor;
    simpletorch::Tensor tensor3 = tensor;
}