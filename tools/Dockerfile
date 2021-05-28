FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      cmake \
      gfortran \
      git \
      intel-mkl \
      python3-appdirs \
      python3-mako \
      python3-numpy \
      python3-pytest \
      python3-pytest-cov \
      python3-pytools \
      python3-pip \
      python3-venv \
      python3-yaml \
      curl \
      && \
    apt-get autoremove --purge -y && \
    apt-get autoclean -y && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

ADD http://icl.utk.edu/projectsfiles/magma/downloads/magma-2.5.4.tar.gz /tmp/
RUN cd /tmp && \
    tar xzf magma-*.tar.gz && \
    rm magma-*.tar.gz && \
    mkdir magma && cd magma && \
    CFLAGS=-I/usr/include/mkl CXXFLAGS=-I/usr/include/mkl cmake ../magma-* -DMKLROOT=/usr -DGPU_TARGET=sm_35 && \
    make -j4 && make install && \
    git clone --recursive --depth 1 -b v1.7.1 https://github.com/pytorch/pytorch /tmp/pytorch && \
    cd /tmp/pytorch && \
    CMAKE_INCLUDE_PATH=/usr/include/mkl TORCH_CUDA_ARCH_LIST=3.5 PYTORCH_BUILD_VERSION=1.7.1 PYTORCH_BUILD_NUMBER=1 python3 setup.py install && \
    cd /tmp && \
    rm -rf magma pytorch

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install \
      'numpy>=1.16' \
      scipy \
      configparser \
      'torchvision==0.8.2' \
      scikit-cuda \
      cupy \
      'tensorflow-gpu>=2.0.0a0' \
      scikit-learn
