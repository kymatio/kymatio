#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=kymatio
#SBATCH --gpus-per-node=1
#SBATCH --time=30
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err



conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda create -q -n $1 python=3.7

conda install -n $1 numpy scipy pytest pytest-cov scikit-learn
conda install -n $1 conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda run -n $1 python3 -m pip install --upgrade pip
conda run -n $1 python3 -m pip install tensorflow

conda run -n $1 python3 -m pip install -r requirements.txt
conda run -n $1 python3 -m pip install -r requirements_optional.txt

conda run -n $1 python3 setup.py develop
conda run -n $1 pytest --cov=kymatio

conda env remove --name $1
dnsqkdnsqjkn