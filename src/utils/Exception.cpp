#include "utils/Exception.h"
#include "utils/StringUtils.h"

namespace c10 {
Error::Error(SourceLocation source_location, std::string msg)
    : Error(
          std::move(msg),
          str("Exception raised from ",
              source_location,
              " (most recent call first):\n")) {}

Error::Error(
    const char* file,
    const uint32_t line,
    const char* condition,
    const std::string& msg,
    const std::string& backtrace,
    const void* caller)
    : Error(
          str("[enforce fail at ",
              c10::StripBasename(file),
              ":",
              line,
              "] ",
              condition,
              ". ",
              msg),
          backtrace,
          caller) {}

Error::Error(std::string msg, std::string backtrace, const void* caller)
  : msg_(std::move(msg)), backtrace_(std::move(backtrace)), caller_(caller) {
  refresh_what();
}

void Error::refresh_what() {
  what_ = compute_what(/*include_backtrace*/ true);
  what_without_backtrace_ = compute_what(/*include_backtrace*/ false);
}

std::string Error::compute_what(bool include_backtrace) const {
  std::ostringstream oss;

  oss << msg_;

  if (context_.size() == 1) {
    // Fold error and context in one line
    oss << " (" << context_[0] << ")";
  } else {
    for (const auto& c : context_) {
      oss << "\n  " << c;
    }
  }

  if (include_backtrace) {
    oss << "\n" << backtrace_;
  }

  return oss.str();
}
}
