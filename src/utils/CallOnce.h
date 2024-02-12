#pragma once

#include "macros/Macros.h"
#include <atomic>
#include <functional>
#include <mutex>
#include <type_traits>
#include <utility>
namespace c10 {

template <typename Flag, typename F, typename... Args>
inline void callOnce(Flag& flag, F&& f, Args&&... args) {
  if (C10_LIKELY(flag.test_once())) {
    return;
  }
  flag.call_once_slow(std::forward<F>(f), std::forward<Args>(args)...);
}

template <typename Functor, typename... Args>
std::enable_if_t<
    std::is_member_pointer_v<std::decay_t<Functor>>,
    typename std::invoke_result_t<Functor, Args...>>
invoke(Functor&& f, Args&&... args) {
  return std::mem_fn(std::forward<Functor>(f))(std::forward<Args>(args)...);
}

template <typename Functor, typename... Args>
std::enable_if_t<
    !std::is_member_pointer_v<std::decay_t<Functor>>,
    typename std::invoke_result_t<Functor, Args...>>
invoke(Functor&& f, Args&&... args) {
  return std::forward<Functor>(f)(std::forward<Args>(args)...);
}

class OnceFlag {
public:
  OnceFlag() noexcept = default;
  OnceFlag(const OnceFlag&) = delete;
  OnceFlag& operator=(const OnceFlag&) = delete;

private:
  template <typename Flag, typename F, typename... Args>
  friend void callOnce(Flag& flag, F&& f, Args&&... args);

  template <typename F, typename... Args>
  void call_once_slow(F&& f, Args&&... args) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (init_.load(std::memory_order_relaxed)) {
      return;
    }
    invoke(std::forward<F>(f), std::forward<Args>(args)...);
    init_.store(true, std::memory_order_release);
  }

  bool test_once() {
    return init_.load(std::memory_order_acquire);
  }

  void reset_once() {
    init_.store(false, std::memory_order_release);
  }

  std::mutex mutex_;
  std::atomic<bool> init_{false};
};
}
