#pragma once

#include "dispatch/DispatchKeySet.h"
#include "dispatch/Dispatcher.h"
namespace c10 {
class KernelFunction final {
public:
  template <class Return, class... Args>
  Return call(const OperatorHandle& opHandle, DispatchKeySet dispatchKeySet, Args... args);
};
}
