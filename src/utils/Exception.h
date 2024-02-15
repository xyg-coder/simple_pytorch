#pragma once

#include "utils/StringUtils.h"
#include "macros/Macros.h"
#include <cstdint>
#include <exception>
#include <string>
#include <vector>
namespace c10 {
class Error : public std::exception {
  // The actual error message.
  std::string msg_;

  // Context for the message (in order of decreasing specificity).  Context will
  // be automatically formatted appropriately, so it is not necessary to add
  // extra leading/trailing newlines to strings inside this vector
  std::vector<std::string> context_;

  // The C++ backtrace at the point when this exception was raised.  This
  // may be empty if there is no valid backtrace.  (We don't use optional
  // here to reduce the dependencies this file has.)
  std::string backtrace_;

  // These two are derived fields from msg_stack_ and backtrace_, but we need
  // fields for the strings so that we can return a const char* (as the
  // signature of std::exception requires).  Currently, the invariant
  // is that these fields are ALWAYS populated consistently with respect
  // to msg_stack_ and backtrace_.
  std::string what_;
  std::string what_without_backtrace_;

  // This is a little debugging trick: you can stash a relevant pointer
  // in caller, and then when you catch the exception, you can compare
  // against pointers you have on hand to get more information about
  // where the exception came from.  In Caffe2, this is used to figure
  // out which operator raised an exception.
  const void* caller_;
public:
  Error(SourceLocation source_location, std::string msg);
  Error(
    const char* file,
    const uint32_t line,
    const char* condition,
    const std::string& msg,
    const std::string& backtrace,
    const void* caller = nullptr);

  // Base constructor
  Error(std::string msg, std::string backtrace, const void* caller = nullptr);
  const char* what() const noexcept override {
    return what_.c_str();
  }
private:
  void refresh_what();
  std::string compute_what(bool include_backtrace) const;
};

class NotImplementedError: public Error {
  using Error::Error;
};

class InvalidArgumentError : public Error {
  using Error::Error;
};

template <typename... Args>
decltype(auto) torchCheckMsgImpl(const char* /*msg*/, const Args&... args) {
  return str(args...);
}
inline const char* torchCheckMsgImpl(const char* msg) {
  return msg;
}
// If there is just 1 user-provided C-string argument, use it.
inline const char* torchCheckMsgImpl(
    const char* /*msg*/,
    const char* args) {
  return args;
}

}

#define TORCH_CHECK_MSG(cond, type, ...)                   \
  (::c10::torchCheckMsgImpl(                               \
      "Expected " #cond                                    \
      " to be true, but got false.  "                      \
      "(Could this error message be improved?  If so, "    \
      "please report an enhancement request to PyTorch.)", \
      ##__VA_ARGS__))

#define C10_UNLIKELY_OR_CONST(e) C10_UNLIKELY(e)

#define C10_THROW_ERROR(err_type, msg) \
  throw ::c10::err_type(               \
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, msg)

#define TORCH_CHECK_WITH_MSG(err_type, cond, type, ...)                  \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {                                 \
    C10_THROW_ERROR(err_type, TORCH_CHECK_MSG(cond, type, __VA_ARGS__)); \
  }

#define TORCH_CHECK_WITH(err_type, cond, ...) \
  TORCH_CHECK_WITH_MSG(err_type, cond, "", __VA_ARGS__)
