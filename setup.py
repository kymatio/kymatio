#!/usr/bin/env python
import os
import shutil
import sys
from setuptools import setup, find_packages
import torch

VERSION = '0.0.1'

long_description = """
Fast CPU/CUDA Scattering Transform implementation for 1D, 2D, and 3D signals

CuPy/PyTorch CUDA and NumPy/PyTorch CUDA implementation
"""

setup_info = dict(
    # Metadata
    name='scattering',
    version=VERSION,
    author=('Edouard Oyallon, Eugene Belilovsky, Sergey Zagoruyko, '
            'Michael Eickenberg, Mathieu Andreux, Georgios Exarchakis, '
            'Louis Thiry, Vincent Lostanlen'),
    author_email=('edouard.oyallon@ens.fr, eugene.belilovsky@inria.fr, '
                  'sergey.zagoruyko@enpc.fr, michael.eickenberg@berkeley.edu, '
                  'mathieu.andreux@ens.fr, georgios.exarchakis@ens.fr, '
                  'louis.thiry@ens.fr, vincent.lostanlen@ens.fr'),
    url='https://github.com/edouardoyallon/pyscatwave',
    description='Fast CPU/CUDA Scattering Transform implementation',
    long_description=long_description,
    license='BSD',

    # Package info
    packages=find_packages(exclude=('test',)),

    zip_safe=True,

    install_requires=[
        'torch',
        'six'
    ]
)

cwd = os.path.dirname(os.path.abspath(__file__))

def create_version_file(cuda):
    global version, cwd
    print('-- Building version ' + VERSION)
    version_path = os.path.join(cwd,'build', 'lib', 'version.py')
    print(version_path)
    with open(version_path, 'w') as f:
        if cuda:
            f.write("CUDA_AVAILABLE = TRUE")
        else:
            f.write("CUDA_AVAILABLE = TRUE")

setup(**setup_info)



CUDA_AVAILABLE = True
if not torch.cuda.is_available():
    CUDA_AVAILABLE = False
try:
    from skcuda import cublas
except:
    CUDA_AVAILABLE = False
try:
    import cupy
except:
    CUDA_AVAILABLE = False

create_version_file(CUDA_AVAILABLE)