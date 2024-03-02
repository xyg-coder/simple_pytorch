#pragma once

#include <ostream>
#include <string>
namespace c10 {
struct OperatorName final {
  OperatorName(std::string&& name, std::string&& overload_name)
    : name(std::move(name)),
      overload_name(std::move(overload_name)) {}

  std::string name;
  std::string overload_name;
};

inline bool operator==(const OperatorName& lhs, const OperatorName& rhs) {
  return lhs.name == rhs.name && lhs.overload_name == rhs.overload_name;
}

inline bool operator!=(const OperatorName& lhs, const OperatorName& rhs) {
  return !operator==(lhs, rhs);
}

std::string toString(const OperatorName& opName);
std::ostream& operator<<(std::ostream&, const OperatorName&);

struct OperatorNameHash {
  size_t operator()(const ::c10::OperatorName& x) const {
    return std::hash<std::string>()(x.name) ^ (~ std::hash<std::string>()(x.overload_name));
  }
};

} // namespace c10
