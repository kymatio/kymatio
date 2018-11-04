Installation
************

We propose two ways to install kymatio: via the `pip` package management system, or directly from the source code.

Recommended
===========

In your comannd line prompt, please run::

    pip install kymatio
    

From source
===========

If you wish to use the latest version of the code::

    git clone https://github.com/kymatio/kymatio
    cd kymatio
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


