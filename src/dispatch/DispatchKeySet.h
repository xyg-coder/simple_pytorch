#pragma once

#include "dispatch/DispatchKey.h"
#include "utils/LlvmMathExtras.h"
#include "utils/Metaprogramming.h"
#include "utils/TypeList.h"
#include <cstdint>
#include <type_traits>

namespace c10 {
// A representation of a set of DispatchKeys. A DispatchKeySet contains both
// "functionality" bits and "backend bits", and every tensor holds its own
// DispatchKeySet. The Dispatcher implements multiple dispatch by grabbing the
// keyset on every input tensor, or’ing them together, and dispatching to a
// specific piece of functionality. The functionality bits are *ordered*. When
// multiple functionality bits are set, we use the highest priority
// functionality. Similarly, multiple backend bits can theoretically be set if
// you call an operator with multiple tensors from difference devices (e.g. CPU
// and CUDA), although support for mixed device dispatch is limited (the only
// kernels that gracefully handle mixed device inputs for now are cuda kernels
// that take in a scalar cpu tensor).

// A representation of a set of DispatchKeys.  A tensor may have multiple
// tensor type ids, e.g., a Variable tensor can also be a CPU tensor; the
// DispatchKeySet specifies what type ids apply.  The internal representation is
// as a 64-bit bit set (this means only 64 tensor type ids are supported).
//
// As mentioned above, DispatchKeys are ordered; thus, we can ask questions like
// "what is the highest priority DispatchKey in the set"?  (The set itself is
// not ordered; two sets with the same ids will always have the ids ordered in
// the same way.)
//
// Note [DispatchKeySet Internal Representation]
// Internally, dispatch keys are packed into 64-bit DispatchKeySet objects
// that get passed around at runtime.
// However, there isn't necessarily a 1-to-1 mapping between bits in the keyset
// and individual dispatch keys.
//
// First: why do we have this distinction, and why not map every dispatch key
// directly to a bit? This is mostly because we have several types of
// functionalities that different backends would like to customize. For example,
// we have:
// - "Dense":     CPU, CUDA, XLA, ... (~12 keys)
// - "Sparse":    SparseCPU, SparseCUDA, ...
// - "Quantized": QuantizedCPU, QuantizedCUDA, QuantizedXLA, ...
// - "Autograd":  AutogradCPU, AutogradCUDA, Autograd XLA, ...
// The problem is that total number of keys grows quadratically with [#
// backends] x [# functionalities], making it very difficult to map each key
// directly to a bit in a bitset without dramatically increasing the size of the
// bitset over time.
//
// The two enums (BackendComponent and DispatchKey) can be divided roughly into
// 5 categories.
//
// (1) "Building block" keys
//    (a) backends: Everything in the BackendComponent enum (e.g. CPUBit,
//    CUDABit) (b) functionalities: (per-backend) functionality-bit DispatchKeys
//    (e.g. AutogradFunctionality, Sparse, Dense)
// (2) "Runtime" keys
//    (a) "non-customizable backends" (e.g. FPGA)
//    (b) "non-customizable functionalities" (e.g. Functionalize)
//    (c) "per-backend instances of customizable functionalities" (e.g. CPU,
//    SparseCPU, AutogradCPU)
// (3) "Alias" DispatchKeys (see Note [Alias Dispatch Keys])
//
// (1) Building block keys always correspond to individual bits in a
// DispatchKeySet. They can also be combined in a DispatchKeySet to form actual
// runtime keys. e.g.
//     auto dense_cpu_ks = DispatchKeySet({DispatchKey::CPUBit,
//     DispatchKey::Dense});
//     // The keyset has the runtime dense-cpu key.
//     dense_cpu_ks.has(DispatchKey::CPU);
//     // And it contains the building block keys too.
//     dense_cpu_ks.has(DispatchKey::CPUBit);
//     dense_cpu_ks.has(DispatchKey::Dense);
//
// Not every backend and not every functionality counts as a "building block
// key". This is mostly to give us more levers to pull in the design space.
// Backend keys and functionality keys that count as "building blocks" will
// contribute to a full cross product of functionality that can be overriden.
//
// For example, right now we have at least 12 "backend" building blocks (CPU,
// CUDA, XLA, ...) and at least 4 "functionality" building blocks (Dense,
// Sparse, Quantized, AutogradFunctionality, ...). These keys together allow
// every dispatcher operator to be customized in up to 12*4 different ways. Each
// of those requires a slot in the operator table of every dispatcher operator.
// Not every piece of functionality necessarily needs to be customizable
// per-backend, and not every backend necessarily needs to be able to customize
// every type of functionality.
//
//
// (2) Every runtime key corresponds directly to a slot in an operator's runtime
// dispatch table, and you can directly register kernels to a runtime dispatch
// key.
//
// For per-backend functionalities like "Dense" or "AutogradFunctionality",
// you can think of the corresponding runtime dispatch keys as "instances" of
// that functionality, per backend. E.g. "CPU", "CUDA", "XLA", etc. are all
// runtime instances of the "Dense" building block key.

// (2a) and (2b) are represented identically in the DispatchKeySet logic:
// - backend-agnostic functionalities (e.g. FuncTorchBatched) are NOT
// customizable per backend.
//   In order to do so, we'd need to promote it to a per-backend functionality
//   "building block" key.
// - non-customizable backends (e.g. FPGA) can NOT customize existing
// functionality like Sparse, Autograd, etc.
//   In order to do so, we'd need to promote it to a backend "building block"
//   key.
//
// In both cases, these keys directly correspond to runtime slots in the
// operator table.
//
//
// (3) "Alias" keys
// See Note [Alias Dispatch Keys]
//
// Final note: for anyone making future changes to the Dispatcher +
// DispatchKeySet internals, there's a closed PR with a basic
// python-implementation of the Dispatcher that might be useful in quickly
// testing out and validating changes. See it at
// https://github.com/pytorch/pytorch/pull/68743

// An undefined tensor is one with an empty tensor type set.
class DispatchKeySet final {
public:
  enum Full { FULL };
  enum Raw { RAW };
  constexpr DispatchKeySet() = default;
  // TODO: revisit this, seems right now the least importantce is using 000
  constexpr DispatchKeySet(Full)
    : repr_((1ULL << (num_functionality_keys - 1)) - 1) {}
  constexpr DispatchKeySet(Raw, uint64_t repr) : repr_(repr) {}
  constexpr DispatchKeySet(uint64_t repr) : repr_(repr) {}
  constexpr DispatchKeySet operator&(DispatchKeySet other) const {
    return DispatchKeySet(repr_ & other.repr_);
  }

    constexpr DispatchKeySet operator|(DispatchKeySet other) const {
    return DispatchKeySet(repr_ | other.repr_);
  }

  // currently we ignore the complicated cases of the dispatchKeyset
  // so dispatchKey is totally the same as dispatchkeyset
  explicit DispatchKeySet(DispatchKey k): repr_(static_cast<uint16_t>(k)) { }

  // returns the DispatchKey of highest priority in the set.
  DispatchKey highestPriorityTypeId() const {
    return static_cast<DispatchKey>(indexOfHighestBit());
  }

  uint8_t indexOfHighestBit() const {
    return 64 - llvm::countLeadingZeros(repr_);
  }

  // returns the index in the operator table of highest priority key in the the keyset
  int getDispatchTableIndexForDispatchKeySet() const {
    return indexOfHighestBit();
  }

  constexpr DispatchKeySet add(DispatchKey t) const {
    return *this | DispatchKeySet(t);
  }

private:
  uint64_t repr_ = 0;
};

inline int getDispatchTableIndexForDispatchKey(DispatchKey k) {
  return DispatchKeySet(k).getDispatchTableIndexForDispatchKeySet();
}


// Given a function type, constructs a function_traits type that drops the first
// parameter type if the first parameter is of type DispatchKeySet. NB:
// DispatchKeySet is currently explicitly hidden from JIT (mainly to avoid
// pushing unnecessary arguments on the stack - see Note [ Plumbing Keys Through
// the Dispatcher] for details). If at any point in the future we need to expose
// this type to JIT, revisit the usage of this type alias.
template <class FuncType>
using remove_DispatchKeySet_arg_from_func = guts::make_function_traits_t<
  typename guts::infer_function_traits_t<FuncType>::return_type,
  typename std::conditional_t<
    std::is_same_v<DispatchKeySet,
      c10::guts::typelist::head_with_default_t<void, typename guts::infer_function_traits_t<FuncType>::parameter_types>>,
    c10::guts::typelist::drop_if_nonempty_t<typename guts::infer_function_traits_t<FuncType>::parameter_types, 1>,
    typename guts::infer_function_traits_t<FuncType>::parameter_types>>;
}
