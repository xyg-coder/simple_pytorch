#include "cuda_ops/EmptyTensor.h"
#include "Context.h"
#include "Device.h"
#include "MemoryFormat.h"
#include "Storage.h"
#include "StorageImpl.h"
#include "Tensor.h"
#include "TensorImpl.h"
#include "TensorOptions.h"
#include "cuda/CudaGuard.h"
#include "utils/ArrayRef.h"
#include "utils/Exception.h"
#include "cuda/CudaAllocator.h"
#include "utils/Typeid.h"
#include "utils/safe_numeric.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace simpletorch::ops {

constexpr uint64_t storage_max() {
  // int64_t and size_t are used somewhat inconsistently throughout ATen.
  // To be safe, storage size calculations must fit in both types.
  constexpr auto int64_max = static_cast<uint64_t>(
      std::numeric_limits<int64_t>::max());
  constexpr auto size_max = static_cast<uint64_t>(
      std::numeric_limits<size_t>::max());
  return std::min(int64_max, size_max);
}

size_t computeStorageNBytesContiguous(
  c10::Int64ArrayRef size_array,
  size_t item_size) {
  
  uint64_t size = 1;
  bool overflow = c10::safe_multiple_uid(
    size_array.begin(), size_array.end(), &size);

  overflow |= c10::mul_overflows(size, item_size, &size);
  overflow |= size > storage_max();

  TORCH_CHECK(!overflow, "Storage size calculation overflowed, size=",
    size)

  return static_cast<size_t>(size);
}

Tensor empty_cuda(
  c10::Int64ArrayRef size_array,
  c10::ScalarType scalarType,
  std::optional<c10::Device> device_opt,
  std::optional<c10::MemoryFormat> memory_format_opt) {

  const auto device = device_or_default(device_opt);
  TORCH_CHECK(device.is_cuda(), "Currently only cuda is supported");
  const auto memory_format = memory_format_or_default(memory_format_opt);
  TORCH_CHECK(memory_format == c10::MemoryFormat::Contiguous,
    "Currently only contiguous memory format is supported");
  
  globalContext().lazyInitCUDA();
  const c10::cuda::CudaGuard guard(device);
  auto* allocator = c10::cuda::cuda_allocator::get();

  for (auto size : size_array) {
    TORCH_CHECK(size > 0, "tensor size cannot be less or equal to 0, size=", size);
  }

  c10::TypeMeta dtype = c10::TypeMeta::fromScalarType(scalarType);
  auto size_bytes = computeStorageNBytesContiguous(
    size_array, dtype.itemsize());
  
  // TODO: use thread-safe shared_ptr to replace
  Storage storage(
    std::make_shared<StorageImpl>(size_bytes, allocator));
  
  auto tensor = Tensor(std::make_shared<TensorImpl>(std::move(storage)));
  if (size_array.size() != 1 || size_array[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size_array);
  }
  return tensor;
}
}
