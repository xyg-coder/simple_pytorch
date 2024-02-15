#include "utils/Exception.h"

void test_check_exception() {
  TORCH_CHECK_WITH(InvalidArgumentError, 1 > 2, "This error is for testing only");
}

void test_throw_exception() {
  TORCH_CHECK_WITH(InvalidArgumentError, 1 < 2, "This error should not be thrown");
  C10_THROW_ERROR(InvalidArgumentError, "This error is for testing only");
}

int main() {
  test_throw_exception();
}
