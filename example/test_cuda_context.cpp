#include "Context.h"
#include <glog/logging.h>

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  simpletorch::globalContext().lazyInitCUDA();

  LOG(INFO) << "number of gpus: " << simpletorch::globalContext().getNumGPUs();
}
