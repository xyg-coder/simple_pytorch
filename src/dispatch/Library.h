#pragma once

#include "dispatch/CppSignature.h"
#include "dispatch/DispatchKey.h"
#include "dispatch/FunctionSchema.h"
#include "dispatch/KernelFunction.h"
#include "dispatch/RegistrationHandleRAII.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
namespace c10 {

/// Represents a C++ function that implements an operator.  Most users won't
/// interact directly with this class, except via error messages: the
/// constructors this function define the set of permissible "function"-like
/// things you can bind via the interface.
///
/// This class erases the type of the passed in function, but durably records
/// the type via an inferred schema for the function.
class CppFunction final {
public:
  template <typename Func>
  explicit CppFunction(
    Func* f,
    std::enable_if_t<c10::guts::is_function_type<Func>::value, std::nullptr_t> = nullptr):
    func_(KernelFunction::makeFromUnboxedRuntimeFunction(f)),
    cpp_signature_(CppSignature::make<Func>()) { }
private:
  KernelFunction func_;
  CppSignature cpp_signature_;

friend class Library;
};

/// This object provides the API for defining operators and providing
/// implementations at dispatch keys.  Typically, a torch::Library
/// is not allocated directly; instead it is created by the
/// TORCH_LIBRARY() or TORCH_LIBRARY_IMPL() macros.
///
/// Most methods on torch::Library return a reference to itself,
/// supporting method chaining.
///
/// ```
/// // Examples:
///
/// TORCH_LIBRARY(torchvision, m) {
///    // m is a torch::Library
///    m.def("roi_align", ...);
///    ...
/// }
///
/// TORCH_LIBRARY_IMPL(aten, XLA, m) {
///    // m is a torch::Library
///    m.impl("add", ...);
///    ...
/// }
/// ```
///
class Library {
public:
  enum Kind {
    DEF,
    IMPL,
  };
  Library(Kind kind, std::string ns, DispatchKey k, const char* file, uint32_t line);

  Library(const Library&) = delete;
  Library& operator=(const Library&) = delete;
  Library(Library&&) = default;
  Library& operator=(Library&&) = default;

  Library& impl_(FunctionSchema&& schema, CppFunction&& f);

  template<typename Func>
  Library& impl(FunctionSchema&& schema, Func&& raw_f) {
    CppFunction f(std::forward<Func>(raw_f));
    return impl_(std::forward<FunctionSchema>(schema), std::move(f));
  }

  Library& def(FunctionSchema&& schema) {
    return _def(std::forward<FunctionSchema>(schema));
  }

  Library& _def(FunctionSchema&& schema);

private:
  Kind kind_;
  std::string ns_;  // namespace
  DispatchKey dispatch_key_;
  const char* file_;
  uint32_t line_;
  std::vector<RegistrationHandleRAII> registars_;
};

class TorchLibraryInit final {
private:
  using InitFn = void(Library&);
  Library lib_;
public:
  TorchLibraryInit(
    Library::Kind kind,
    InitFn* fn,
    const char* ns,
    c10::DispatchKey k,
    const char* file,
    uint32_t line): lib_(kind, ns, k, file, line) {
      fn(lib_);
    };
};
} // namespace c10

#define TORCH_LIBRARY(ns, m)                                          \
  static void TORCH_LIBRARY_INIT_##ns(c10::Library&);                 \
  static const c10::TorchLibraryInit TORCH_LIBRARY_static_init_##ns(  \
    c10::Library::DEF,                                                \
    &TORCH_LIBRARY_INIT_##ns,                                         \
    #ns,                                                              \
    std::nullopt,                                                     \
    __FILE__,                                                         \
    __LINE__);                                                        \
  void TORCH_LIBRARY_init_##ns(c10::Library& m)


/// Macro for defining a function that will be run at static
/// initialization time to define operator overrides for dispatch key
/// `k` (must be an unqualified enum member of c10::DispatchKey) in
/// namespace `ns` (must be a valid C++ identifer, no quotes).  Use this
/// macro when you want to implement a preexisting set of custom
/// operators on a new dispatch key (e.g., you want to provide CUDA
/// implementations of already existing operators).  One common usage
/// pattern is to use TORCH_LIBRARY() to define schema for all new
/// operators you want to define, and then use several
/// TORCH_LIBRARY_IMPL() blocks to provide implementations of the
/// operator for CPU, CUDA and Autograd.
///
/// In some cases, you need to define something that applies to all namespaces,
/// not just one namespace (usually a fallback).  In that case, use the reserved
/// namespace _, e.g.,
///
/// ```
/// TORCH_LIBRARY_IMPL(_, XLA, m) {
///    m.fallback(xla_fallback);
/// }
/// ```
///
/// Example usage:
///
/// ```
/// TORCH_LIBRARY_IMPL(myops, CPU, m) {
///   // m is a torch::Library; methods on it will define
///   // CPU implementations of operators in the myops namespace.
///   // It is NOT valid to call torch::Library::def()
///   // in this context.
///   m.impl("add", add_cpu_impl);
/// }
/// ```
///
/// If ``add_cpu_impl`` is an overloaded function, use a
/// ``static_cast`` to specify which overload you want
/// (by providing the full type).
///
// NB: if the dispatch key is not whitelisted, we simply omit the Library
// call entirely
#define TORCH_LIBRARY_IMPL(ns, k, m) _TORCH_LIBRARY_IMPL(ns, k, m, C10_UID)

#define _TORCH_LIBRARY_IMPL(ns, k, m, uid)                                                                \
  static void C10_CONCATENATE(TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(c10::Library&);                 \
  static const c10::TorchLibraryInit C10_CONCATENATE(TORCH_LIBRARY_IMPL_static_init_##ns##_##k##_, uid)(  \
    c10::Library::IMPL,                                                                                   \
    &C10_CONCATENATE(TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid),                                         \
    #ns,                                                                                                  \
    c10::DispatchKey::k,                                                              \
    __FILE__,                                                                                             \
    __LINE__);                                                                                            \
  void C10_CONCATENATE(TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(c10::Library& m)
