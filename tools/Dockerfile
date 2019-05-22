FROM nvidia/cuda:9.2-devel

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      python3-scipy \
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
      configparser \
      torchvision \
      scikit-cuda \
      cupy
