#pragma once

#include "dispatch/DispatchKeySet.h"
namespace c10 {
struct DispatchKeyExtractor {
  template<class... Args>
  DispatchKeySet getDispatchKeySetUnboxed(const Args&... args) const;
};
};
