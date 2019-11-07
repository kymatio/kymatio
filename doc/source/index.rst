
.. |pic1| image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg

.. |pic2| image:: https://img.shields.io/badge/python-3.5%2C%203.6%2C%203.7-blue.svg

.. |picDL| image:: https://pepy.tech/badge/kymatio

.. |pic3| image:: https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png
          :width: 30px
          :target: https://github.com/kymatio/kymatio

.. |pic4| image:: https://avatars3.githubusercontent.com/u/50278?s=200&v=4
          :width: 30px
          :target: https://twitter.com/KymatioWavelets

.. |flatiron| image:: _static/FL_Full_Logo_Mark_Small.png
          :width: 300px
          :target: https://www.simonsfoundation.org/flatiron

.. |ens| image:: https://www.ens.fr/sites/default/files/inline-images/logo.jpg
          :width: 300px
          :target: https://www.ens.fr/

.. scattering documentation master file, created by
   sphinx-quickstart on Tue Oct  2 23:41:09 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



Kymatio
*******

|pic2|  |pic1| |picDL|

|pic3|  |pic4|


Kymatio is a Python module for computing wavelet and scattering transforms.

It is built on top of standard Python libraries, such as NumPy, PyTorch, and TensorFlow. Our framework allows the use of
high-level frontends to directly plug the scattering network to your application. It also incorporates different backends
which allow for low-level tuning of the code. For instance, the PyTorch API includes a fast CUDA backend via cupy and skcuda.

Use Kymatio if you need a library that:

* integrates wavelet scattering in a deep learning architecture,
* supports 1-D, 2-D, and 3-D scattering transforms,
* differentiable transforms for applications in generative modeling, reconstruction, and more!
* runs seamlessly on CPU and GPU hardware, with major deep learning APIs
  (PyTorch and TensorFlow).

A brief intro to wavelet scattering is provided in :ref:`user-guide`. For a
list of publications see
`Publications <https://www.di.ens.fr/data/publications>`_. If you use this package, please cite the following paper:

Andreux M., Angles T., Exarchakis G., Leonarduzzi R., Rochette G., Thiry L., Zarka J., Mallat S., Andén J.,
Belilovsky E., Bruna J., Lostanlen V., Hirn M. J., Oyallon E., Zhang S., Cella C., Eickenberg M. (2019). **Kymatio:
Scattering Transforms in Python.** `arXiv preprint <https://arxiv.org/abs/1812.11214>`_.

We wish to thank the Scientific Computing Core at the Flatiron Institute for the use of their computing resources for testing.

|flatiron|

We would also like to thank École Normale Supérieure for their support.

|ens|

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
