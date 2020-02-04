#!/bin/bash
set -e

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
else
    pip install --upgrade pytest
    pip install pytest-cov
    pip3 install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
    pip install torchvision
fi

pip install 'tensorflow>=2.0.0a' \
            scikit-learn

pip install -r requirements.txt
python setup.py develop
