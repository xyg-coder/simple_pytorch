#pragma once

namespace simpletorch {
class Tensor {
protected:
    // for testing right now
    int value;
public:
    Tensor(int value);
    int get_value();
};
}
