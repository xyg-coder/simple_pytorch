#pragma once

#include "dispatch/DispatchKeySet.h"
#include "dispatch/FunctionSchema.h"
namespace c10 {

// Take a DispatchKeySet for a Tensor and determine what the actual dispatch
// DispatchKey should be, taking into account TLS, and skipping backends which
// fall through.
static inline DispatchKeySet computeDispatchKeySet(
  DispatchKeySet ks,
  DispatchKeySet key_mask) {
  return (ks & key_mask);
}

struct DispatchKeyExtractor {
  // compare to the original implementation, we ignore the arguments
  template<class... Args>
  DispatchKeySet getDispatchKeySetUnboxed(const Args&... args) const {
    return non_fall_through_keys_;
  }
  void registerSchema(const FunctionSchema& schema) {}
  void deregisterSchema() {}
private:
  DispatchKeySet non_fall_through_keys_;
};
};
