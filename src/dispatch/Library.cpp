#include "dispatch/Library.h"
#include "dispatch/Dispatcher.h"
#include "dispatch/FunctionSchema.h"
#include <utility>

namespace c10 {

namespace {
std::string debugString(const char* file, uint32_t line) {
    return c10::str("registered at ", file, ":", line);
  }
}

Library::Library(Kind kind, std::string ns, DispatchKey k, const char* file, uint32_t line):
  kind_(kind), ns_(ns), dispatch_key_(k), file_(file), line_(line) { }


Library& Library::impl_(FunctionSchema&& schema, CppFunction&& f) {
  registars_.emplace_back(Dispatcher::singleton().registerImpl(
    schema.operatorName(),
    dispatch_key_,
    f.func_,
    f.cpp_signature_,
    debugString(file_, line_)));
  return *this;
}

Library& Library::_def(FunctionSchema&& schema) {
  registars_.emplace_back(Dispatcher::singleton().registerDef(
    std::move(schema), debugString(file_, line_)));
  return *this;
}

} //namespace c10
