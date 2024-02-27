#pragma once

#include "dispatch/OperatorName.h"
namespace c10 {
// every implementation should have one schema defined here
struct FunctionSchema {
const OperatorName& operatorName() const {
  return name_;
}
private:
  OperatorName name_;

};
};
