#pragma once

#include "dispatch/DispatchKeySet.h"
#include <cstring>
#include <string>
#include <type_traits>
#include <typeindex>
#include <utility>
namespace c10 {
class CppSignature final {
public:
  CppSignature(const CppSignature&) = default;
  CppSignature(CppSignature&&) noexcept = default;
  CppSignature& operator=(const CppSignature&) = default;
  CppSignature& operator=(CppSignature&&) noexcept = default;

  template <class FuncType>
  static CppSignature make() {
    using decayed_func_type = typename c10::remove_DispatchKeySet_arg_from_func<std::decay_t<FuncType>>::func_type;
    return CppSignature(std::type_index(typeid(decayed_func_type)));
  }

  std::string name() const {
    return std::string(signature_.name());
  }

  friend bool operator==(const CppSignature& lhs, const CppSignature& rhs) {
    if (lhs.signature_ == rhs.signature_) {
        return true;
    }
    // Without RTLD_GLOBAL, the type_index comparison could yield false because
    // they point to different instances of the RTTI data, but the types would
    // still be the same. Let's check for that case too.
    // Note that there still is a case where this might not work, i.e. when
    // linking libraries of different compilers together, they might have
    // different ways to serialize a type name. That, together with a missing
    // RTLD_GLOBAL, would still fail this.
    if (0 == strcmp(lhs.signature_.name(), rhs.signature_.name())) {
        return true;
    }

    return false;
  }

  friend bool operator !=(const CppSignature& lhs, const CppSignature& rhs) {
    return !operator==(lhs, rhs);
  }

private:
  explicit CppSignature(std::type_index signature):
    signature_(std::move(signature)) {}
  std::type_index signature_;
};
};
