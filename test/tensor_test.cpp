#include <gtest/gtest.h>
#include "CpuAllocator.h"
#include "Storage.h"
#include "StorageImpl.h"
#include "Tensor.h"
#include "TensorImpl.h"

// Demonstrate some basic assertions.
TEST(TensorTest, BasicAssertions) {
    c10::NaiveCpuAllocator allocator;

    simpletorch::Storage storage(
        std::make_shared<simpletorch::StorageImpl>(100 * sizeof(int), &allocator));
    simpletorch::Tensor tensor(std::make_shared<simpletorch::TensorImpl>(std::move(storage)));
    simpletorch::Tensor tensor2 = tensor;
    simpletorch::Tensor tensor3 = tensor;
  // Expect equality.
  EXPECT_EQ(tensor.get_unsafe_data(), tensor2.get_unsafe_data());
  EXPECT_EQ(tensor.get_unsafe_data(), tensor3.get_unsafe_data());
}
