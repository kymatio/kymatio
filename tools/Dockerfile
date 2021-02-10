FROM nvidia/cuda:10.0-devel-ubuntu18.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      libcudnn7 \
      python3-appdirs \
      python3-mako \
      python3-pytest \
      python3-pytest-cov \
      python3-pytools \
      python3-pip \
      python3-venv \
      curl \
      && \
    apt-get autoremove --purge -y && \
    apt-get autoclean -y && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

RUN pip3 install \
      'numpy>=1.16' \
      scipy \
      configparser \
      'torch<1.3' \
      torchvision \
      scikit-cuda \
      cupy \
      'tensorflow-gpu>=2.0.0a0' \
      scikit-learn
