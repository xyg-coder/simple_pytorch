#include <gtest/gtest.h>
#include "Allocator.h"
#include "CpuAllocator.h"
#include "Storage.h"
#include "StorageImpl.h"
#include "Tensor.h"
#include "TensorImpl.h"
#include "ops/EmptyOps.h"
#include "utils/ArrayRef.h"

// Demonstrate some basic assertions.
TEST(AllocationTest, TensorCopy) {
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

void helper_for_copy(int &count) {
  c10::DeleteFnPtr delete_fn = [&count](void *data) {
    count++;
    c10::deleteNaiveCpuData(data);
  };
  c10::NaiveCpuAllocator allocator(delete_fn);
  simpletorch::Storage storage(
      std::make_shared<simpletorch::StorageImpl>(100 * sizeof(int), &allocator));
  simpletorch::Tensor tensor(std::make_shared<simpletorch::TensorImpl>(std::move(storage)));
  simpletorch::Tensor tensor2 = tensor;
  simpletorch::Tensor tensor3 = tensor;
}

// Demonstrate some basic assertions.
TEST(AllocationTest, DataDestructor) {

  int count = 0;
  c10::DeleteFnPtr delete_fn = [&count](void *data) {
    count++;
    c10::deleteNaiveCpuData(data);
  };
  c10::NaiveCpuAllocator allocator(delete_fn);

  simpletorch::Storage storage(
      std::make_shared<simpletorch::StorageImpl>(100 * sizeof(int), &allocator));
  simpletorch::Tensor* tensor = new simpletorch::Tensor(std::make_shared<simpletorch::TensorImpl>(std::move(storage)));
  simpletorch::Tensor* tensor2 = new simpletorch::Tensor(*tensor);
  simpletorch::Tensor* tensor3 = new simpletorch::Tensor(*tensor2);
  delete tensor;
  EXPECT_EQ(count, 0);
  delete tensor2;
  EXPECT_EQ(count, 0);
  delete tensor3;
  EXPECT_EQ(count, 1);
  helper_for_copy(count);
  EXPECT_EQ(count, 2);
}

TEST(AllocationTest, EmptyTensorSize) {
  std::array<int64_t, 2> size_array {5, 6};
	c10::Int64ArrayRef size(size_array);
	simpletorch::Tensor tensor = simpletorch::ops::empty::call(
		size, c10::ScalarType::Double, std::nullopt, std::nullopt);
  c10::Int64ArrayRef result_size = tensor.get_sizes();
  EXPECT_EQ(result_size.size(), size.size());
  EXPECT_EQ(result_size[0], 5);
  EXPECT_EQ(result_size[1], 6);
}
