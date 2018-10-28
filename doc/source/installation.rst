Installation
************

For installing the software, we propose two solutions: via pip or from the sources.

Recommended
===========

In your bash, run::

    pip install scattering_transform

From source
===========

If you desire to use the latest version of the code::

    git clone https://github.com/scattering_transform/scattering_transform
    cd scattering_transform
    pip install -r requirements.txt
    python setup.py install

Optimizing GPU acceleration
===========================

In order to benefit from cuda optimization, you can also install skcuda and cupy::

    pip install cupy skcuda

and then refer to :ref:`backend-story` for more details about the use of our optimized cuda kernels.

Developer
=========

??????????????????????????????Use `python setup.py develop`????????????????????


