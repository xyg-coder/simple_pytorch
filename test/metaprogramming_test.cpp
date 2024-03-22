#include "utils/Metaprogramming.h"
#include "utils/TypeList.h"
#include "utils/TypeTraits.h"
#include <gtest/gtest.h>
#include <type_traits>
#include <utility>
#include "dispatch/DispatchKeySet.h"

using namespace c10::guts;

namespace {

int dummy_function(int, int) {
  return 0;
}

struct dummy_functor {
  int operator()(int, int) {
    return 0;
  }
};

//////////
// type_traits.h
//////////
static_assert(c10::guts::is_instantiation_of<
  typelist::typelist, typelist::typelist<int, int>>::value, "");

static_assert(!c10::guts::is_instantiation_of<
  typelist::typelist, int>::value, "");

static_assert(
  std::is_same<int (int, int),
    strip_class<decltype(&dummy_functor::operator())>::type>::value, "");
  
static_assert(
  !std::is_same<double (int, int),
    strip_class<decltype(&dummy_functor::operator())>::type>::value, "");

class AddValue {  
    int value;  
public:  
    AddValue(int v) : value(v) {}  
  
    int operator()(int x) const {  
        return x + value;  
    }  
};

class NonFunctor {  
    int value;  
};

static_assert(is_functor<AddValue>::value, "");
static_assert(!is_functor<NonFunctor>::value, "");

static_assert(is_function_type<int (int, int)>::value, "");
static_assert(!is_function_type<int>::value, "");
//////////////////////
// typelist.h
//////////////////////
static_assert(!false_t<int>::value, "");


static_assert(std::is_same<typelist::head_with_default_t<void, typelist::typelist<int, double>>, int>::value, "");
static_assert(std::is_same<typelist::head_with_default_t<void, typelist::typelist<>>, void>::value, "");

static_assert(std::is_same<typelist::element<0, typelist::typelist<int, double>>::type, int>::value, "");
static_assert(std::is_same<typelist::element<1, typelist::typelist<int, double>>::type, double>::value, "");
static_assert(std::is_same<typelist::element<2, typelist::typelist<int, double, bool>>::type, bool>::value, "");

static_assert(typelist::size<typelist::typelist<int, double>>::value == 2);
static_assert(typelist::size<typelist::typelist<>>::value == 0);

static_assert(
  typelist::arg_list_compare<
    typelist::take_elements<
      typelist::typelist<int, bool, double, int>, 1, std::index_sequence<0, 1>>::type,
    typelist::typelist<bool, double>>::value);
  static_assert(
  !typelist::arg_list_compare<
    typelist::take_elements<
      typelist::typelist<int, bool, double, int>, 1, std::index_sequence<0, 1>>::type,
    typelist::typelist<bool>>::value);
  
static_assert(
  typelist::arg_list_compare<
    typelist::take<
      typelist::typelist<int, bool, double, int>, 3>::type,
    typelist::typelist<int, bool, double>>::value);

static_assert(
  typelist::arg_list_compare<
    typelist::take<
      typelist::typelist<int, bool, double, int>, 4>::type,
    typelist::typelist<int, bool, double, int>>::value);

static_assert(
  typelist::arg_list_compare<
    typelist::drop_if_nonempty<
      typelist::typelist<int, bool, double, int>, 4>::type,
    typelist::typelist<>>::value);

static_assert(
  typelist::arg_list_compare<
    typelist::drop_if_nonempty<
      typelist::typelist<int, bool, double, int>, 5>::type,
    typelist::typelist<>>::value);

static_assert(
  typelist::arg_list_compare<
    typelist::drop_if_nonempty<
      typelist::typelist<int, bool, double, int>, 1>::type,
    typelist::typelist<bool, double, int>>::value);

static_assert(typelist::arg_list_compare<int, int>::value);
static_assert(!typelist::arg_list_compare<int, bool>::value);
static_assert(typelist::arg_list_compare<typelist::typelist<int, double>, typelist::typelist<int, double>>::value);
static_assert(!typelist::arg_list_compare<typelist::typelist<int, double>, typelist::typelist<int, bool>>::value);
static_assert(!typelist::arg_list_compare<typelist::typelist<int, double>, typelist::typelist<int>>::value);

//////////////////////
// Metaprogramming.h
//////////////////////
static_assert(
  std::is_same<make_function_traits<int, typelist::typelist<bool, bool>>::type::return_type,
    int>::value
);

static_assert(
  std::is_same<make_function_traits<int, typelist::typelist<bool, bool>>::type::func_type,
    int(bool, bool)>::value
);

static_assert(
  typelist::arg_list_compare<
    make_function_traits<int, typelist::typelist<bool, bool>>::type::parameter_types,
    typelist::typelist<bool, bool>>::value);

int add(int a, int b) {
  return a + b;
}

static_assert(
  std::is_same<infer_function_traits<int(bool, bool)>::type::return_type,
    int>::value
);

static_assert(
  std::is_same<infer_function_traits<decltype(add)>::type::return_type,
    int>::value
);

static_assert(
  std::is_same<infer_function_traits<int(bool, bool)>::type::func_type,
    int(bool, bool)>::value
);

static_assert(
  typelist::arg_list_compare<
    infer_function_traits<int(bool, bool)>::type::parameter_types,
    typelist::typelist<bool, bool>>::value);
  
static_assert(
  std::is_same<infer_function_traits<int(*)(bool, bool)>::type::return_type,
    int>::value
);

static_assert(
  std::is_same<infer_function_traits<int(*)(bool, bool)>::type::func_type,
    int(bool, bool)>::value
);

static_assert(
  typelist::arg_list_compare<
    infer_function_traits<int(*)(bool*, bool)>::type::parameter_types,
    typelist::typelist<bool*, bool>>::value);

// test functor
struct MockFunctor {
  int operator() (int a, int b) {
    return a + b;
  }
};

static_assert(typelist::arg_list_compare<
    infer_function_traits<MockFunctor>::type::parameter_types,
    typelist::typelist<int, int>>::value);

static_assert(
  std::is_same<infer_function_traits<MockFunctor>::type::return_type,
    int>::value);

// test tuple args
static_assert(
  std::is_same<std::tuple_element_t<0, infer_function_traits<MockFunctor>::type::ArgsTuple>, int>::value);

static_assert(
  std::is_same<std::tuple_element_t<0, infer_function_traits<decltype(add)>::type::ArgsTuple>, int>::value);

//////////////////////
// DispatchKeySet.h
//////////////////////
static_assert(
  std::is_same<
    c10::remove_DispatchKeySet_arg_from_func<int(int, int)>::func_type, int(int, int)>::value);

static_assert(
  std::is_same<
    c10::remove_DispatchKeySet_arg_from_func<int(c10::DispatchKeySet, int)>::func_type, int(int)>::value);

static_assert(
  std::is_same<
    c10::remove_DispatchKeySet_arg_from_func<int(c10::DispatchKeySet, c10::DispatchKeySet)>::func_type, int(c10::DispatchKeySet)>::value);

static_assert(
  std::is_same<
    c10::remove_DispatchKeySet_arg_from_func<int()>::func_type, int()>::value);

static_assert(std::is_same<typelist::head_with_default_t<
  void, infer_function_traits_t<int(c10::DispatchKeySet)>::parameter_types>, c10::DispatchKeySet>::value);
} // anonymous namespace
