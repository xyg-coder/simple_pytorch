#pragma once

#include "StorageImpl.h"
#include "utils/intrusive_ptr.h"

namespace simpletorch {

class Storage {

protected:
    c10::intrusive_ptr<StorageImpl> storage_impl_;
};
}