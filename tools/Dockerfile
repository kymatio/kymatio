FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      libcudnn8 \
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

RUN pip3 install \
      scipy \
      configparser \
      torchvision \
      scikit-cuda \
      cupy \
      'tensorflow-gpu>=2.0.0a0' \
      scikit-learn
