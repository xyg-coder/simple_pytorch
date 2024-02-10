# Simple torch
## quick test
```
mkdir build
cd build
cmake ..
cmake --build .
GLOG_logtostderr=1 ./TestTensorExecutable ## this way we can output the log
ctest ## run unittests
```