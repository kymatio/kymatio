Installation
************

There are two ways to install kymatio: via the `pip` package management system, or directly from source.


Recommended
===========

In a shell, please run::

    pip install kymatio
    

From source
===========

To install from source, first install `PyTorch <https://pytorch.org/>`_. This is most easily achieved inside the Anaconda enviroment by running::

    conda install pytorch torchvision -c pytorch

We then obtain the latest version of Kymatio::

    git clone https://github.com/kymatio/kymatio

Finally, the package is installed by running::

    cd kymatio
    pip install -r requirements.txt
    python setup.py install


Optimizing GPU acceleration
===========================

To improve performance with an optimized CUDA implementation, you may install optional CUDA packages by running::

    pip install -r requirements_optional_cuda.txt

To enable this implementation, see the :ref:`backend-story` section.


Developer
=========

For developers, we recommend performing a development install. The steps are
same as installing from source (see above), but with the last line replaced
by::

    python setup.py develop
