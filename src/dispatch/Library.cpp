#include "dispatch/Library.h"

namespace c10 {
Library::Library(Kind kind, std::string ns, DispatchKey k, const char* file, uint32_t line):
  kind_(kind), ns_(ns), dispatch_key_(k), file_(file), line_(line) { }

} //namespace c10
