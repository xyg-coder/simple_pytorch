#pragma once

#include <cstdint>
namespace c10 {

/// ArrayRef - Represent a constant reference to an array (0 or more elements
/// consecutively in memory), i.e. a start pointer and a length.  It allows
/// various APIs to take consecutive elements easily and conveniently.
///
/// This class does not own the underlying data, it is expected to be used in
/// situations where the data resides in some other buffer, whose lifetime
/// extends past that of the ArrayRef. For this reason, it is not in general
/// safe to store an ArrayRef.
///
/// This is intended to be trivially copyable, so it should be passed by
/// value.
template<typename T>
class ArrayRef final {

};

using Int64ArrayRef = ArrayRef<int64_t>;
}
