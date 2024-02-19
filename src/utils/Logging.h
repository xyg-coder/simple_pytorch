#pragma once

#include <glog/logging.h>
#include "utils/StringUtils.h"

#define LOG_INFO(...) LOG(INFO) << c10::str(__VA_ARGS__);

#define LOG_ERROR(...) LOG(ERROR) << c10::str(__VA_ARGS__);
