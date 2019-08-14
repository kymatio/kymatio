#!/bin/bash

sudo apt-get update
if [[ $CONDA == "1" ]]; then
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    hash -r
    conda config --set always_yes yes --set changeps1 no
    conda update -q conda
    conda info -a

    conda create -q -n test-environment python=$TRAVIS_PYTHON_VERSION numpy scipy pytest pytest-cov
    source activate test-environment
    conda install -c pytorch pytorch-cpu
    pip install tensorflow
else
    pip install --upgrade pytest
    pip install pytest-cov
    if [[ "$TRAVIS_PYTHON_VERSION" == "3.5" ]]; then
        pip install https://download.pytorch.org/whl/cpu/torch-1.0.0-cp35-cp35m-linux_x86_64.whl
    elif [[ "$TRAVIS_PYTHON_VERSION" == "3.6" ]]; then
        pip install https://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
    elif [[ "$TRAVIS_PYTHON_VERSION" == "3.7" ]]; then
        pip install https://download.pytorch.org/whl/cpu/torch-1.0.0-cp37-cp37m-linux_x86_64.whl
    fi
    pip install torchvision
    pip install tensorflow
fi

pip install -r requirements.txt
python setup.py develop
