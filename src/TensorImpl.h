#pragma once

#include "Storage.h"
#include "utils/intrusive_ptr.h"

namespace simpletorch {
class TensorImpl  : public c10::instrusive_ptr_target {
protected:
    Storage storage;
};
}