#!/bin/bash
#SBATCH -N1
#SBATCH --ntasks=1 --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH -C T4
#SBATCH --mem=32G
#SBATCH --mail-user=your-email@your-domain.com
#SBATCH --mail-type=ALL

module load cuda10.1/toolkit/10.1.105/conda

EXPORT CONDA_ENV=test

conda install -n ${CONDA_ENV} numpy scipy pytest pytest-cov
conda install -n ${CONDA_ENV} conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda run -n ${CONDA_ENV} python3 -m pip install --upgrade pip
conda run -n ${CONDA_ENV} python3 -m pip install "tensorflow>=2.0.0a"
conda run -n ${CONDA_ENV} python3 -m pip install scikit-learn

conda run -n ${CONDA_ENV} python3 -m pip install -r requirements.txt
conda run -n ${CONDA_ENV} python3 -m pip install -r requirements_optional.txt

conda run -n ${CONDA_ENV} python3 setup.py develop
conda run -n ${CONDA_ENV} pytest --cov=kymatio