#pragma once

namespace c10 {

namespace detail {
// constructor tag used by intrusive_ptr constructor
struct DontIncreaseRfcount {};

template <class TTarget>
struct intrusive_target_default_null_type final {
    static constexpr TTarget* singleton() noexcept {
        return nullptr;
    }
};
}

class instrusive_ptr_target {

};

template<class TTarget, class NullType=detail::intrusive_target_default_null_type<TTarget>>
class intrusive_ptr final {

private:
    TTarget* target_;
};

}
