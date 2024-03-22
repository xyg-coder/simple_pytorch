#pragma once

#include <cstddef>
#include <utility>
namespace c10::guts {
#if defined(__cpp_lib_apply) && !defined(__CUDA_ARCH__)

template <class F, class Tuple>
__host__ __device__ inline constexpr decltype(auto) apply(F&& f, Tuple&& t) {
  return std::apply(std::forward<F>(f), std::forward<Tuple>(t));
}

#else
// Implementation from http://en.cppreference.com/w/cpp/utility/apply (but
// modified)
// TODO This is an incomplete implementation of std::apply, not working for
// member functions.
template <class F, class Tuple, std::size_t... INDEX>
__host__ __device__ constexpr decltype(auto) apply_impl(
    F&& f,
    Tuple&& t,
    std::index_sequence<INDEX...>)
{
  return std::forward<F>(f)(std::get<INDEX>(std::forward<Tuple>(t))...);
}

template <class F, class Tuple>
__host__ __device__ constexpr decltype(auto) apply(F&& f, Tuple&& t) {
  return apply_impl(
      std::forward<F>(f),
      std::forward<Tuple>(t),
      std::make_index_sequence<
          std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}
#endif
} // namespace c10::guts
