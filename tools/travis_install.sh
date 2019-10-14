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
    conda install pytorch torchvision cpuonly -c pytorch
    pip install 'tensorflow>=2.0.0'
else
    pip install --upgrade pytest
    pip install pytest-cov
    if [[ "$TRAVIS_PYTHON_VERSION" == "3.5" ]]; then
        pip install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
    elif [[ "$TRAVIS_PYTHON_VERSION" == "3.6" ]]; then
        pip install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
    elif [[ "$TRAVIS_PYTHON_VERSION" == "3.7" ]]; then
        pip install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
    fi
    pip install 'tensorflow>=2.0.0'
fi

pip install -r requirements.txt
python setup.py develop
