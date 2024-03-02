#include "dispatch/DispatchKey.h"

namespace c10 {
std::ostream& operator<<(std::ostream& os, DispatchKey dispatchKey) {
  return os << toString(dispatchKey);
}

const char* toString(DispatchKey t) {
switch (t) {
    case DispatchKey::Autograd:
      return "AutoGrad"; 
    case DispatchKey::CUDA:
      return "CUDA";
    default:
      return "Undefined";
  }
}

}
