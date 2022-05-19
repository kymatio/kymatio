FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      libcudnn8 \
      python3-pytest \
      python3-pytest-cov \
      python3-pip \
      python3-venv \
      curl \
      && \
    apt-get autoremove --purge -y && \
    apt-get autoclean -y && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

RUN pip3 install --upgrade \
      numpy \
      scipy \
      configparser \
      torchvision \
      scikit-cuda \
      cupy \
      'tensorflow-gpu>=2.0.0a0' \
      scikit-learn
