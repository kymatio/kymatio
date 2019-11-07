Installation
************

There are two ways to install kymatio: via the `pip` package management system, or directly from source. As of Kymatio 0.2,
the PyTorch frontend is the default frontend. In Kymatio 0.3, this will be
changed to NumPy instead.


Recommended
===========

In a shell, please run::

    pip install kymatio
    

From source
===========

To install from source, install the latest version of Kymatio::

    git clone https://github.com/kymatio/kymatio

Finally, the package is installed by running::

    cd kymatio
    pip install -r requirements.txt
    python setup.py install


Optimizing GPU acceleration
===========================

To improve performance on PyTorch with an optimized CUDA implementation, you may install optional CUDA packages by
running::

    pip install scikit-cuda cupy

To enable this implementation, see the :ref:`backend-story` section.


Developer
=========

For developers, we recommend performing a development install. The steps are
same as installing from source (see above), but with the last line replaced
by::

    python setup.py develop
