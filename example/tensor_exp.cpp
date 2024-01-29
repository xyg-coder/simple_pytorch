# include <iostream>
# include "Tensor.h"

int main() {
    simpletorch::Tensor tensor(123);
    std::cout << tensor.get_value() << std::endl;
}
