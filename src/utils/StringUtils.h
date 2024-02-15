# pragma once

#include <sstream>
#include <string>
namespace c10 {
std::string StripBasename(const std::string& full_path);

// this struct will convert T to const& T
template <typename T>
struct CanonicalizeStrTypes {
  using type = const T&;
};

struct SourceLocation {
  const char* function;
  const char* file;
  uint32_t line;
};

std::ostream& operator<<(std::ostream& out, const SourceLocation& loc);

template <typename T>
inline std::ostream& _str(std::ostream& ss, const T& t) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  ss << t;
  return ss;
}

template <typename T, typename... Args>
inline std::ostream& _str(std::ostream& ss, const T& t, const Args&... args) {
  return _str(_str(ss, t), args...);
}

template <typename... Args>
struct _str_wrapper final {
  static std::string call(const Args&... args) {
    std::ostringstream ss;
    _str(ss, args...);
    return ss.str();
  }
};

// Convert a list of string-like arguments into a single string.
template <typename... Args>
inline decltype(auto) str(const Args&... args) {
  return _str_wrapper<
      typename CanonicalizeStrTypes<Args>::type...>::call(args...);
}
}
