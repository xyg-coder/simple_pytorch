# Simple torch
This is one toy project to learn how pytorch cpp kernel works. I tried to do these
* Learn how pytorch implements some feature in the cpp code
* Implement one simplified version. So you can find a lot of codes are very similar. However, I tried to reduce the complexity as much as possible.
## Implemented features
* Tensor, tensorbody, tensor storage, etc
* Gpu memory initialization
* Function dispatch.
* Some gpu kernels. Current I only implemented the gpu version for `fill`, `zeros`.
## Future features if I have time
* Matrix calculation
* More complicated gpu kernels
* Gradient backward
## Build the dev env
I used nvidia cuda docker to set up env quickly.
* `docker pull nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04`
* `docker run --gpus all --name pytorch-dev-container-run-indefinitely 5d846bce3f98  tail -f /dev/null` this command will sets up dev container and run indefinitely.
* Another window, `docker exec -it pytorch-dev-container-run-indefinitely /bin/bash`
* Inside the docker
  ```
  apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  openssh-client \
  build-essential \
  ca-certificates \
  ccache \
  cmake \
  curl \
  git \
  libjpeg-dev \
  libpng-dev && \
  rm -rf /var/lib/apt/lists/*
  ```
* Install the conda
  ```
  apt-get install wget
  apt-get update
  apt-get install wget
  wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.3.1-0-Linux-x86_64.sh
  bash Miniconda3-py38_23.3.1-0-Linux-x86_64.sh
  ```
* `source ~/.bashrc` should init conda env already
After these steps, you should be ready to start c++ env.
### version `GLIBCXX_3.4.30` not found
https://stackoverflow.com/questions/73317676/importerror-usr-lib-aarch64-linux-gnu-libstdc-so-6-version-glibcxx-3-4-30

## Make commands
```
make all # build all targets
make target TARGET=xxx # build some specific target. You can find the targets in the CMakeLists.txt
make logrun TARGET=xxx # run some specific target with log.info enabled
make gtest # run all the tests under test/
```

## Examples
You can see the use cases under `/example` or `/test`. They show some cases this project already covers. Hopefully I have time to implement more simplified versions in the future.
