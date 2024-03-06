#pragma once

#include "dispatch/OperatorName.h"
#include "utils/Exception.h"
#include <ostream>
#include <sstream>
namespace c10 {

struct Argument {};

// every implementation should have one schema defined here
struct FunctionSchema {
enum FunctionSchmaId {
  EMPTY,
  // below for testing
  TEST,
  ALLOCATOR,
};

FunctionSchema() = delete;
FunctionSchema(FunctionSchmaId&& schemaId, OperatorName&& opName)
  :name_(std::move(opName)), schemaId_(std::move(schemaId)) {}

const OperatorName& operatorName() const {
  return name_;
}

const FunctionSchmaId& schemaId() const {
  return schemaId_;
}
private:
  OperatorName name_;
  FunctionSchmaId schemaId_;
};

inline std::string toString(const FunctionSchema::FunctionSchmaId& schemaId) {
  switch (schemaId) {
    case FunctionSchema::FunctionSchmaId::TEST:
      return "TEST"; 
    default:
      TORCH_CHECK(false, "Invalid schemaId: ", schemaId);
  }
  return "";
}

inline std::ostream& operator<<(std::ostream& out, const FunctionSchema& schema) {
  return out << "FunctionSchema, operatorName=("
    << schema.operatorName() << "), functionSchemaId=("
    << toString(schema.schemaId()) << ")";
}

inline std::string toString(const FunctionSchema& schema) {
  std::ostringstream str;
  str << schema;
  return str.str();
}
};
