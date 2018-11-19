#!/usr/bin/env python
import csv
import imp
import os
import shutil
import sys
from setuptools import setup, find_packages

DESCRIPTION = 'Wavelet scattering transforms in Python with GPU acceleration'
VERSION = imp.load_source('kymatio.version', 'kymatio/version.py').version
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()
URL = 'https://kymatio.github.io'
LICENSE = 'BSD-3-Clause'


# Parse requirements.txt
with open('requirements.txt', 'r') as f:
    REQUIREMENTS = [row[0] for row in csv.reader(f, delimiter='\n')]


setup_info = dict(
    # Metadata
    name='kymatio',
    version=VERSION,
    author=('Edouard Oyallon, Eugene Belilovsky, Sergey Zagoruyko, '
            'Michael Eickenberg, Mathieu Andreux, Georgios Exarchakis, '
            'Louis Thiry, Vincent Lostanlen, Joakim Andén, '
            'Tomás Angles, Gabriel Huang, Roberto Leonarduzzi'),
    author_email=('edouard.oyallon@centralesupelec.fr, belilove@iro.umontreal.ca, '
                  'sergey.zagoruyko@inria.fr, michael.eickenberg@berkeley.edu, '
                  'mathieu.andreux@ens.fr, georgios.exarchakis@ens.fr, '
                  'louis.thiry@ens.fr, vincent.lostanlen@nyu.edu, janden@flatironinstitute.org, '
                  'tomas.angles@ens.fr, gabriel.huang@umontreal.ca, roberto.leonarduzzi@ens.fr'),
    url=URL,
    download_url='https://github.com/kymatio/kymatio/releases',
    classifiers=['Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Natural Language :: English',
                 'Operating System :: MacOS',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: POSIX :: Linux',
                 'Programming Language :: Python :: 3.4',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Topic :: Multimedia :: Graphics :: 3D Modeling',
                 'Topic :: Multimedia :: Sound/Audio :: Analysis',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Scientific/Engineering :: Chemistry',
                 'Topic :: Scientific/Engineering :: Image Recognition',
                 'Topic :: Scientific/Engineering :: Information Analysis',
                 'Topic :: Scientific/Engineering :: Mathematics',
                 'Topic :: Scientific/Engineering :: Physics',
                 'Topic :: Software Development :: Libraries :: Python Modules',
                 ],
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    license=LICENSE,
    packages=find_packages(exclude=('test',)),
    install_requires=REQUIREMENTS,
    zip_safe=True,
)

setup(**setup_info)
