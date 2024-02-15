#include "utils/StringUtils.h"
#include <string>

namespace c10 {
std::string StripBasename(const std::string& full_path) {
  const std::string separators("/");
  size_t pos = full_path.find_last_of(separators);
  if (pos != std::string::npos) {
    return full_path.substr(pos + 1, std::string::npos);
  } else {
    return full_path;
  }
}

std::ostream& operator<<(std::ostream& out, const SourceLocation& loc) {
  out << loc.function << " at " << loc.file << ":" << loc.line;
  return out;
}
}
