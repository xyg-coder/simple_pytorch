#pragma once

#include <string>
namespace c10 {
struct OperatorName final {
  OperatorName(std::string&& name, std::string&& overload_name)
    : name(std::move(name)),
      overload_name(std::move(overload_name)) {}

  std::string name;
  std::string overload_name;
};
};
