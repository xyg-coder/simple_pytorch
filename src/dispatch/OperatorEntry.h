#pragma once

#include "dispatch/CppSignature.h"
#include "dispatch/DispatchKey.h"
#include "dispatch/DispatchKeySet.h"
#include "dispatch/DispatcherKeyExtractor.h"
#include "dispatch/FunctionSchema.h"
#include "dispatch/KernelFunction.h"
#include "dispatch/OperatorName.h"
#include "utils/Exception.h"
#include <array>
#include <memory>
#include <optional>
#include <string>
namespace c10 {

struct AnnotatedSchema final {
  AnnotatedSchema(FunctionSchema&& s, std::string&& d): schema_(std::move(s)), debug_(std::move(d)) {};
  FunctionSchema schema_;
  std::string debug_;
};

struct AnnotatedKernel final {
  AnnotatedKernel(KernelFunction&& k,
    std::unique_ptr<FunctionSchema>&& s, std::string&& d)
    :kernel(std::move(k)), inferred_function_schema(std::move(s)),
    debug(std::move(d)) {}
  KernelFunction kernel;
  std::unique_ptr<FunctionSchema> inferred_function_schema;
  std::string debug;
};


// Internal data structure that records information about a specific operator.
// It's not part of the public API; typically, users will interact with
// OperatorHandle instead.
//
// Concurrent writes to OperatorEntry are protected by the GLOBAL Dispatcher
// lock (this is important because some methods in OperatorEntry access
// dispatcher state)
class OperatorEntry {
public:
  explicit OperatorEntry(OperatorName&& operator_name);

  OperatorEntry(const OperatorEntry&) = delete;
  OperatorEntry(OperatorEntry&&) noexcept = delete;
  OperatorEntry& operator=(const OperatorEntry&) = delete;
  OperatorEntry& operator=(OperatorEntry&&) noexcept = delete;

  const FunctionSchema& schema() const {
    TORCH_CHECK(schema_.has_value(),
      "Tried to access the schema for ", name_, " which doesn't have a schema");
    return schema_->schema_;
  }
  
  const std::string& debug() const {
    TORCH_CHECK(schema_.has_value(),
      "Tried to access the schema for ", name_, " which doesn't have a schema");
    return schema_->debug_;
  }

  bool hasSchema() const {
    return schema_.has_value();
  }

  const DispatchKeyExtractor& dispatchKeyExtractor() const {
    return dispatch_key_extractor_;
  }

  const OperatorName& operatorName() const {
    return name_;
  }

  void assertSignatureIsCorrect(const CppSignature& call_signature, bool has_symint) const;

  template<class FuncType>
  inline void assertSignatureIsCorrect();

  const KernelFunction& lookup(DispatchKeySet ks) const;

  void registerSchema(FunctionSchema&&, std::string&& debug);
  void deregisterSchema();

  using AnnotatedKernelContainer = std::list<AnnotatedKernel>;
  using AnnotatedKernelContainerIterator = std::list<AnnotatedKernel>::iterator;

  AnnotatedKernelContainer registerKernel(
    const Dispatcher& dispatcher,
    std::optional<DispatchKey> dispatchKey,
    KernelFunction kernelFunction,
    std::optional<CppSignature> cpp_signature,
    std::unique_ptr<FunctionSchema> inferred_func_schema,
    std::string debug);

  void deregisterKernel_(
    const Dispatcher& dispatcher,
    std::optional<DispatchKey> dispatch_key,
    AnnotatedKernelContainerIterator kernel);

private:
  OperatorName name_;
  std::optional<AnnotatedSchema> schema_;
  std::array<KernelFunction, num_runtime_entries> dispatch_table_;
  DispatchKeyExtractor dispatch_key_extractor_;
};
}
