
.. |pic1| image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg

.. |pic2| image:: https://img.shields.io/badge/python-3.5%2C%203.6%2C%203.7-blue.svg

.. |picDL| image:: https://pepy.tech/badge/kymatio

.. |piccodecov| image:: https://codecov.io/gh/kymatio/kymatio/branch/master/graph/badge.svg

.. |pic3| image:: https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png
          :width: 30px
          :target: https://github.com/kymatio/kymatio

.. |pic4| image:: https://avatars3.githubusercontent.com/u/50278?s=200&v=4
          :width: 30px
          :target: https://twitter.com/KymatioWavelets

.. scattering documentation master file, created by
   sphinx-quickstart on Tue Oct  2 23:41:09 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



Kymatio
*******

|pic2|  |pic1| |picDL| |piccodecov|

|pic3|  |pic4|


Kymatio is a Python module for computing wavelet and scattering transforms.

It is built on top of PyTorch, but also has a fast CUDA backend via cupy and
skcuda.

Use kymatio if you need a library that:

* integrates wavelet scattering in a deep learning architecture,
* supports 1-D, 2-D, and 3-D scattering transforms
* differentiable transforms for applications in generative modeling, reconstruction and more!
* runs seamlessly on CPU and GPU hardware.

A brief intro to wavelet scattering is provided in :ref:`user-guide`. For a
list of publications see
`Publications <https://www.di.ens.fr/data/publications>`_.

.. include:: quickstart.rst

.. toctree::
   :maxdepth: 2

   installation
   userguide
   developerguide
   codereference
   gallery_1d/index
   gallery_2d/index
   gallery_3d/index
