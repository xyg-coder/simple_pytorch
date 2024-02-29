#include "dispatch/OperatorEntry.h"
#include "dispatch/CppSignature.h"
#include "dispatch/DispatchKeySet.h"
#include "dispatch/FunctionSchema.h"
#include "macros/Macros.h"
#include "utils/Exception.h"
#include <optional>

namespace c10 {
void OperatorEntry::assertSignatureIsCorrect(const CppSignature& call_signature) const {
  if (C10_UNLIKELY(cpp_signature_.has_value() && cpp_signature_->signature != call_signature)) {
    reportSignatureError(call_signature, cpp_signature_.value());
  }
}

void OperatorEntry::reportSignatureError(const CppSignature& call_signature, const CppSignatureWithDebug& saved_signature) const {
  TORCH_CHECK(false,
        "\nTried to access or call an operator with a wrong signature.\n",
        "  operator: ", (schema_.has_value() ? toString(schema_->schema_) : toString(name_)), "\n",
        "    ", (schema_.has_value() ? schema_->debug_ : "unknown debug info"), "\n",
        "  correct signature:  ", saved_signature.signature.name(), "\n",
        "    ", saved_signature.debug, "\n",
        "  accessed/called as: ", call_signature.name(), "\n",
        "This likely happened in a call to OperatorHandle::typed<Return (Args...)>(). ",
        "Please make sure that the function signature matches the signature in the operator registration call."
  );
};

void OperatorEntry::reportError(DispatchKey dispatchKey) const {
  TORCH_CHECK_WITH(NotImplementedError, false, "Could not run '", name_, "' with arguments.",
          " This could be because "
          "the operator doesn't exist for this backend, or was omitted during ",
          "the selective/custom build process (if using custom build).");
}

void OperatorEntry::registerSchema(FunctionSchema&& schema, std::string&& debug) {
  TORCH_CHECK(!schema_.has_value());
  dispatch_key_extractor_.registerSchema(schema);
  schema_ = AnnotatedSchema(std::move(schema), std::move(debug));
}

void OperatorEntry::deregisterSchema() {
  TORCH_CHECK(schema_.has_value());  
  schema_ = std::nullopt;
  dispatch_key_extractor_.deregisterSchema();
}

void OperatorEntry::registerKernel(
  const Dispatcher& dispatcher,
  DispatchKey dispatchKey,
  KernelFunction kernelFunction,
  std::optional<CppSignature> cpp_signature,
  // TODO: do we really need inferred_func_schema?
  std::unique_ptr<FunctionSchema> inferred_func_schema,
  std::string debug) {
  
  if (cpp_signature.has_value()) {
    if (cpp_signature_.has_value()) {
      TORCH_CHECK(cpp_signature.value() == cpp_signature_->signature,
        "\nMismatch in kernel registration.\n",
        "  operator: ", (schema_.has_value() ? toString(schema_->schema_) : toString(name_)), "\n",
        "    ", (schema_.has_value() ? schema_->debug_ : "unknown debug info"), "\n",
        "  correct signature:  ", cpp_signature_->signature.name(), "\n",
        "    ", cpp_signature_->debug, "\n",
        "  accessed/called as: ", cpp_signature->name(), "\n");
    } else {
      cpp_signature_ = CppSignatureWithDebug{
        *cpp_signature,
        debug,
        dispatchKey};
    }
  }

  auto dispatch_table_index = getDispatchTableIndexForDispatchKey(dispatchKey);
  TORCH_CHECK(!dispatch_table_[dispatch_table_index].kernel.isValidUnboxed(),
    "Kernel function for dispatch_table_index=", dispatch_table_index, " is already initialized.\n",
    " Currently we don't support function overriding yet.");
  dispatch_table_[dispatch_table_index] = 
    AnnotatedKernel {
      std::move(kernelFunction),
      std::move(inferred_func_schema),
      std::move(debug)};
}

void OperatorEntry::deregisterKernel_(
  const Dispatcher& dispatcher,
  DispatchKey dispatch_key) {
  
  auto dispatch_table_index = getDispatchTableIndexForDispatchKey(dispatch_key);
  AnnotatedKernel kernel = dispatch_table_[dispatch_table_index];
  TORCH_CHECK(kernel.kernel.isValidUnboxed(),
    "Trying to deregister one invalid kernel for dispatch_key=",
    dispatch_key, ", debug_string=", kernel.debug);
  dispatch_table_[dispatch_table_index] = AnnotatedKernel();
}
}
