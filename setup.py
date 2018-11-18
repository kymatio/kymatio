#!/usr/bin/env python
import os
import shutil
import sys
from setuptools import setup, find_packages

VERSION = '0.0.1'

long_description = """
Fast CPU/CUDA Scattering Transform implementation for 1D, 2D, and 3D signals

CuPy/PyTorch CUDA and NumPy/PyTorch CUDA implementation
"""

setup_info = dict(
    # Metadata
    name='kymatio',
    version=VERSION,
    author=('Edouard Oyallon, Eugene Belilovsky, Sergey Zagoruyko, '
            'Michael Eickenberg, Mathieu Andreux, Georgios Exarchakis, '
            'Louis Thiry, Vincent Lostanlen, Joakim Anden, '
            'Tomas Angles, Gabriel Huang, Roberto Leonarduzzi'),
    author_email=('edouard.oyallon@ens.fr, eugene.belilovsky@inria.fr, '
                  'sergey.zagoruyko@enpc.fr, michael.eickenberg@berkeley.edu, '
                  'mathieu.andreux@ens.fr, georgios.exarchakis@ens.fr, '
                  'louis.thiry@ens.fr, vincent.lostanlen@nyu.edu, janden@flatironinstitute.org, '
                  'tomas.angles@ens.fr, gabriel.huang@ens.fr, roberto.leonarduzzi@ens.fr'),
    url='https://github.com/edouardoyallon/pyscatwave',
    description='Fast CPU/CUDA Scattering Transform implementation',
    long_description=long_description,
    license='BSD',

    # Package info
    packages=find_packages(exclude=('test',)),

    zip_safe=True,

    install_requires=[
        'torch'
    ]
)

setup(**setup_info)
