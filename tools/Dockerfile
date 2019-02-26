FROM ubuntu:bionic

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && \
    add-apt-repository ppa:graphics-drivers/ppa && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      libnvidia-compute-410 \
      nvidia-cuda-toolkit \
      python3-scipy \
      python3-appdirs \
      python3-pytest \
      python3-pip \
      python3-venv \
      && \
    apt-get autoremove --purge -y && \
    apt-get autoclean -y && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*
