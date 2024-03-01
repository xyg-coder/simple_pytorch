#include "dispatch/Dispatcher.h"
#include "dispatch/FunctionSchema.h"
#include "dispatch/OperatorName.h"
#include "dispatch/RegistrationHandleRAII.h"

int main() {
  c10::RegistrationHandleRAII raii = c10::Dispatcher::singleton()
    .registerDef(
      c10::FunctionSchema(
        c10::FunctionSchema::TEST,
        c10::OperatorName("TEST", "")
      ),
      "test-debug"
    );
}
