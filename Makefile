clean:
	mkdir -p build
	rm -rf build/*

all:
	mkdir -p build
	cd build && cmake ..
	make -C build

target:
	mkdir -p build
	cd build && cmake ..
	make -C build ${TARGET}

logrun:
	GLOG_logtostderr=1 ./build/${TARGET}

gtest:
	cd build && ctest