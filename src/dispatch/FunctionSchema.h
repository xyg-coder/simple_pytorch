#pragma once

#include "dispatch/OperatorName.h"
#include <ostream>
#include <sstream>
namespace c10 {

struct Argument {};

// every implementation should have one schema defined here
struct FunctionSchema {
const OperatorName& operatorName() const {
  return name_;
}
private:
  OperatorName name_;
};

inline std::ostream& operator<<(std::ostream& out, const FunctionSchema& schema) {

}

inline std::string toString(const FunctionSchema& schema) {
  std::ostringstream str;
  str << schema;
  return str.str();
}

};
