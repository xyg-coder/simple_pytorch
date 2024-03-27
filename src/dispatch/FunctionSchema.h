#pragma once

#include "dispatch/OperatorName.h"
#include "utils/Exception.h"
#include <ostream>
#include <sstream>
namespace c10 {

#define FUNCTION_SCHEMA_LIST_ITER(_)   \
  _(EMPTY, "EMPTY")                    \
  _(FILL, "FILL")                    \
  _(TEST, "TEST")                      \
  _(ALLOCATOR, "ALLOCATOR")            \

struct Argument {};

// every implementation should have one schema defined here
struct FunctionSchema {
enum FunctionSchmaId {
  #define FETCH_ELEMENT(n, s) n,
  FUNCTION_SCHEMA_LIST_ITER(FETCH_ELEMENT)
  #undef FETCH_ELEMENT
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
    #define BUILD_SWITCH(n, s)  \
    case(FunctionSchema::FunctionSchmaId::n): \
    return s;
    FUNCTION_SCHEMA_LIST_ITER(BUILD_SWITCH)
    #undef BUILD_SWITCH
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
