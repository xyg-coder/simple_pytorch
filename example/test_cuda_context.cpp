#include "Context.h"
#include "utils/Logging.h"
#include <glog/logging.h>

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  simpletorch::globalContext().lazyInitCUDA();

  LOG_INFO("number of gpus: ", simpletorch::globalContext().getNumGPUs());
}
