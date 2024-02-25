#pragma once

#include "utils/TypeList.h"
#include "utils/TypeTraits.h"
#include <type_traits>
namespace c10::guts {
/**
 * Access information about result type or arguments from a function type.
 * Example:
 * using A = function_traits<int (float, double)>::return_type // A == int
 * using A = function_traits<int (float, double)>::parameter_types::tuple_type
 * // A == tuple<float, double>
 */
template <class Func>
struct function_traits {
  static_assert(
      !std::is_same_v<Func, Func>,
      "In function_traits<Func>, Func must be a plain function type.");
};

template <class Result, class... Args>
struct function_traits<Result(Args...)> {
  using func_type = Result(Args...);
  using return_type = Result;
  using parameter_types = typelist::typelist<Args...>;
  static constexpr auto number_of_parameters = sizeof...(Args);
};

template <typename Result, typename Arglist>
struct make_function_traits {
  static_assert(std::false_type::value,
    "In guts::make_function_traits<Result, TypeList>, the ArgList argument must be typelist<...>.");
};

template <typename Result, typename... Args>
struct make_function_traits<Result, typelist::typelist<Args...>> {
  using type = function_traits<Result(Args...)>;
};

template <typename Result, typename ArgList>
using make_function_traits_t = typename make_function_traits<Result, ArgList>::type;

/**
 * infer_function_traits: creates a `function_traits` type for a simple
 * function (pointer) or functor (lambda/struct). Currently does not support
 * class methods.
 */
template<typename Functor>
struct infer_function_traits {
  using type = function_traits<c10::guts::strip_class_t<decltype(&Functor::operator())>>;
};

template<typename Result, typename... Args>
struct infer_function_traits<Result (*)(Args...)> {
  using type = function_traits<Result(Args...)>;
};

template<typename Result, typename... Args>
struct infer_function_traits<Result(Args...)> {
  using type = function_traits<Result(Args...)>;
};
}
