#!/usr/bin/env python
import os
import shutil
import sys
from setuptools import setup, find_packages

VERSION = '0.0.1'

long_description = """
Fast CPU/CUDA Scattering implementation

CuPy/PyTorch CUDA and NumPy/PyTorch CUDA implementation
"""

setup_info = dict(
    # Metadata
    name='scatwave',
    version=VERSION,
    author='Edouard Oyallon, Eugene Belilovsky, Sergey Zagoruyko',
    author_email='edouard.oyallon@ens.fr, eugene.belilovsky@inria.fr, sergey.zagoruyko@enpc.fr',
    url='https://github.com/edouardoyallon/pyscatwave',
    description='Fast CPU/CUDA Scattering implementation',
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

setup(**setup_info)
