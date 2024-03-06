#pragma once

#include <type_traits>
namespace c10::guts {
/**
 * strip_class: helper to remove the class type from pointers to `operator()`
 */
template <typename T>
struct strip_class {};

template <typename Class, typename Result, typename... Args>
struct strip_class<Result (Class::*)(Args...)> {
  using type = Result(Args...);
};

template <typename Class, typename Result, typename... Args>
struct strip_class<Result (Class::*)(Args...) const> {
  using type = Result(Args...);
};

template <typename T>
using strip_class_t = typename strip_class<T>::type;

/**
 * is_function_type<T> is true_type iff T is a plain function type (i.e.
 * "Result(Args...)")
 */
template <class T>
struct is_function_type : std::false_type {};
template <class Result, class... Args>
struct is_function_type<Result(Args...)> : std::true_type {};
template <class T>
using is_function_type_t = typename is_function_type<T>::type;


template <class Functor, class Enable = void>
struct is_functor : std::false_type {};
template <class Functor>
struct is_functor<
    Functor,
    std::enable_if_t<is_function_type<
        strip_class_t<decltype(&Functor::operator())>>::value>>
    : std::true_type {};


/**
 * is_instantiation_of<T, I> is true_type iff I is a template instantiation of T
 * (e.g. vector<int> is an instantiation of vector) Example:
 *    is_instantiation_of_t<vector, vector<int>> // true
 *    is_instantiation_of_t<pair, pair<int, string>> // true
 *    is_instantiation_of_t<vector, pair<int, string>> // false
 */
template <template <class...> class Template, class T>
struct is_instantiation_of : std::false_type {};
template <template <class...> class Template, class... Args>
struct is_instantiation_of<Template, Template<Args...>> : std::true_type {};
template <template <class...> class Template, class T>
using is_instantiation_of_t = typename is_instantiation_of<Template, T>::type;
}
