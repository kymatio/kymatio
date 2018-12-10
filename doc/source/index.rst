.. scattering documentation master file, created by
   sphinx-quickstart on Tue Oct  2 23:41:09 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Kymatio:
********

.. image:: https://img.shields.io/badge/python-3.6-blue.svg

.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg

Kymatio is a Python module for computing wavelet and scattering transforms.

It is built on top of PyTorch, but also has a fast CUDA backend via cupy and
skcuda.

Use kymatio if you need a library that:

* integrates wavelet scattering in a deep learning architecture,
* supports 1-D, 2-D, and 3-D wavelets, and
* runs seamlessly on CPU and GPU hardware.

A brief intro to wavelet scattering is provided in :ref:`user-guide`. For a
list of publications see
`Publications <https://www.di.ens.fr/data/publications>`_.

.. include:: quickstart.rst

.. toctree::
   :maxdepth: 2
   :caption: Wavelet Scattering in PyTorch

   installation
   userguide
   developerguide
   codereference
   gallery_1d/index
   gallery_2d/index
   gallery_3d/index
