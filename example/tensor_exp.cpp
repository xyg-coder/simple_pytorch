#include "CpuAllocator.h"
#include "ScalarType.h"
#include "Storage.h"
#include "StorageImpl.h"
#include "Tensor.h"
#include "TensorImpl.h"
#include "cuda_ops/EmptyTensor.h"
#include "utils/ArrayRef.h"
#include <array>
#include <cstdint>
#include <memory>
#include <glog/logging.h>
#include <optional>

simpletorch::Tensor get_cuda_tensor() {
	std::array<int64_t, 2> size_array {5, 6};
	c10::Int64ArrayRef size(size_array);
	return simpletorch::ops::empty_cuda(
		size, c10::ScalarType::Double, std::nullopt, std::nullopt);
}

int main(int argc, char* argv[]) {
	google::InitGoogleLogging(argv[0]);


	c10::NaiveCpuAllocator allocator;

	simpletorch::Storage storage(
		std::make_shared<simpletorch::StorageImpl>(100 * sizeof(int), &allocator));
	simpletorch::Tensor tensor(std::make_shared<simpletorch::TensorImpl>(std::move(storage)));
	simpletorch::Tensor tensor2 = tensor;
	simpletorch::Tensor tensor3 = tensor;
	simpletorch::Tensor tensor4 = get_cuda_tensor();
}
