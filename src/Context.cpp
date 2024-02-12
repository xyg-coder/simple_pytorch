#include "Context.h"

namespace simpletorch {
Context& globalContext() {
  static Context globalContext_;
  return globalContext_;
}
}
