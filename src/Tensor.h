#pragma once

#include "TensorImpl.h"
#include "utils/intrusive_ptr.h"

namespace simpletorch {
class Tensor {
protected:
    c10::intrusive_ptr<TensorImpl> tensor_impl_;
};
}
