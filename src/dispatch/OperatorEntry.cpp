#include "dispatch/OperatorEntry.h"
#include "dispatch/CppSignature.h"
#include "macros/Macros.h"
#include "utils/Exception.h"

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
}
