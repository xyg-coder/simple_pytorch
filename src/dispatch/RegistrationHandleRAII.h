#pragma once

#include <functional>
namespace c10 {
class RegistrationHandleRAII final {
public:
  explicit RegistrationHandleRAII(std::function<void()>&& on_destruction)
    : on_destruction_(std::move(on_destruction)) {}
  ~RegistrationHandleRAII() {
    if (on_destruction_) {
      on_destruction_();
    }
  }

  RegistrationHandleRAII(const RegistrationHandleRAII&) = delete;
  RegistrationHandleRAII& operator=(const RegistrationHandleRAII&) = delete;

  RegistrationHandleRAII(RegistrationHandleRAII&& rhs) noexcept
      : on_destruction_(std::move(rhs.on_destruction_)) {
    rhs.on_destruction_ = nullptr;
  }

  RegistrationHandleRAII& operator=(RegistrationHandleRAII&& rhs) noexcept {
    if (this == &rhs) {
      return *this;
    }
    on_destruction_ = std::move(rhs.on_destruction_);
    rhs.on_destruction_ = nullptr;
    return *this;
  }
private:
  std::function<void()> on_destruction_;
};
}
