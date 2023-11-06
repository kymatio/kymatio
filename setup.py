#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
from setuptools import setup, find_packages

# Constants
DISTNAME = 'kymatio'
DESCRIPTION = 'Wavelet scattering transforms in Python with GPU acceleration'
URL = 'https://www.kymat.io'
LICENSE = 'BSD-3-Clause'


# Parse description
with open('README.md', encoding='utf8') as f:
    README = f.read().split('\n')
    LONG_DESCRIPTION = '\n'.join([x for x in README if not x[:3] == '[!['])


# Parse version.py
kymatio_version_spec = importlib.util.spec_from_file_location(
    'kymatio_version', 'kymatio/version.py')
kymatio_version_module = importlib.util.module_from_spec(kymatio_version_spec)
kymatio_version_spec.loader.exec_module(kymatio_version_module)
VERSION = kymatio_version_module.version


# Parse requirements.txt
with open('requirements.txt', 'r') as f:
    REQUIREMENTS = f.read().split('\n')


setup_info = dict(
    # Metadata
    name=DISTNAME,
    version=VERSION,
    author=('Edouard Oyallon, Eugene Belilovsky, Sergey Zagoruyko, '
            'Michael Eickenberg, Mathieu Andreux, Georgios Exarchakis, '
            'Louis Thiry, Vincent Lostanlen, Joakim Andén, '
            'Tomás Angles, Gabriel Huang, Roberto Leonarduzzi'),
    author_email=('edouard.oyallon@lip6.fr, belilove@iro.umontreal.ca, '
                  'sergey.zagoruyko@inria.fr, michael.eickenberg@flatironinstitute.org, '
                  'mathieu.andreux@ens.fr, georgios.exarchakis@ens.fr, '
                  'louis.thiry@ens.fr, vincent.lostanlen@nyu.edu, janden@kth.se, '
                  'tomas.angles@ens.fr, gabriel.huang@umontreal.ca, roberto.leonarduzzi@ens.fr'),
    url=URL,
    download_url='https://github.com/kymatio/kymatio/releases',
    project_urls={
        'Documentation': 'https://www.kymat.io/codereference.html',
        'Source': 'https://github.com/kymatio/kymatio/',
        'Tracker': 'https://github.com/kymatio/kymatio/issues',
        'Authors': 'https://github.com/kymatio/kymatio/blob/master/AUTHORS.md'
    },
    classifiers=['Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Natural Language :: English',
                 'Operating System :: MacOS',
                 'Operating System :: POSIX :: Linux',
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
    python_requires='>=3.8',
    license=LICENSE,
    packages=find_packages(exclude=('test',)),
    install_requires=REQUIREMENTS,
    zip_safe=True,
)

setup(**setup_info)
