# include "Tensor.h"

namespace simpletorch {
int Tensor::get_value() {
    return this->value;
}

Tensor::Tensor(int value): value(value) {}
}