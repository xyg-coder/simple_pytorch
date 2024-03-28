#pragma once

#include "utils/TypeTraits.h"
#include <algorithm>
#include <type_traits>
#include <cstddef>
#include <utility>

namespace c10::guts {


template <class... T>
struct false_t: std::false_type {};

namespace typelist {
template <class... Items>
struct typelist final {
public:
  typelist() = delete;
};

template <class Default, class TypeList>
struct head_with_default {
  using type = Default;
};

template <class Default, class Head, class ...Tail>
struct head_with_default<Default, typelist<Head, Tail...>> {
  using type = Head;
};

template <class Default, class Typelist>
using head_with_default_t = typename head_with_default<Default, Typelist>::type;

/**
 * Returns the N-th element of a type list.
 * Example:
 * int == element_t<1, typelist<float, int, char>>
 */
template <size_t Index, class TypeList>
struct element : std::false_type {};

template <class Head, class... Tail>
struct element <0, typelist<Head, Tail...>> {
  using type = Head;
};

/// Error case, we have an index but ran out of types! It will only be selected
/// if `Ts...` is actually empty!
template <size_t Index, class... Ts>
struct element <Index, typelist<Ts...>> : std::false_type { };

template <size_t Index, class Head, class... Ts>
struct element <Index, typelist<Head, Ts...>> : element<Index - 1, typelist<Ts...>> {};

/// Convenience alias.
template <size_t Index, class TypeList>
using element_t = typename element<Index, TypeList>::type;

template <class TypeList>
struct size : std::false_type {};

template <class... types>
struct size <typelist<types...>> final {
  static constexpr size_t value = sizeof...(types);
};

/**
 * Take/drop a number of arguments from a typelist.
 * Example:
 *   typelist<int, string> == take_t<typelist<int, string, bool>, 2>
 *   typelist<bool> == drop_t<typelist<int, string, bool>, 2>
 */
template <class TypeList, size_t offset, class IndexSequence>
struct take_elements : std::false_type {};

template <class TypeList, size_t offset, size_t... Indices>
struct take_elements<TypeList, offset, std::index_sequence<Indices...>> final {
  using type = typelist<element_t<offset + Indices, TypeList>...>;
};

template <class TypeList, size_t num>
struct take final {
  static_assert(is_instantiation_of<typelist, TypeList>::value,
    "In typelist::take<T, num>, the T argument must be typelist<...>.");
  static_assert(num <= size<TypeList>::value,
    "Tried to typelist::take more elements than there are in the list");
  using type = typename take_elements<TypeList, 0, std::make_index_sequence<num>>::type;
};

/**
 * Like drop, but returns an empty list rather than an assertion error if `num`
 * is larger than the size of the TypeList.
 * Example:
 *   typelist<> == drop_if_nonempty_t<typelist<string, bool>, 2>
 *   typelist<> == drop_if_nonempty_t<typelist<int, string, bool>, 3>
 */
template <class TypeList, size_t num>
struct drop_if_nonempty final {
  static_assert(
      is_instantiation_of<typelist, TypeList>::value,
      "In typelist::drop<T, num>, the T argument must be typelist<...>.");
  using type = typename take_elements<
      TypeList,
      std::min(num, size<TypeList>::value),
      std::make_index_sequence<
          size<TypeList>::value - std::min(num, size<TypeList>::value)>>::type;
};
template <class TypeList, size_t num>
using drop_if_nonempty_t = typename drop_if_nonempty<TypeList, num>::type;

// compare 2 lists of args
template <class T1, class T2>
struct arg_list_compare : std::is_same<T1, T2> {};

template <class T1, class... TypeList1, class T2>
struct arg_list_compare <typelist<T1, TypeList1...>, T2> : std::false_type {};

template <class T1, class... TypeList1, class T2, class... TypeList2>
struct arg_list_compare <typelist<T1, TypeList1...>, typelist<T2, TypeList2...>> {
  static constexpr bool value = std::is_same_v<T1, T2> && arg_list_compare<typelist<TypeList1...>, typelist<TypeList2...>>::value;
};

// check if T1 is equal to T2 or inside T2 if T2 is a typelist
template <class T1, class T2>
struct arg_type_inside: std::is_same<T1, T2> {};

template <class T1, class T2, class... TypeList1>
struct arg_type_inside <T1, typelist<T2, TypeList1...>> {
  static constexpr bool value = std::is_same_v<T1, T2> || arg_type_inside<T1, typelist<TypeList1...>>::value;
};

} // namespace typelist
} // namespace c10::guts
