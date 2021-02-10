.. _dev-guide:

Information for developers
**************************

(GitHub Workflow)
=================

Kymatio implements the scattering transform for different frontends (currently ``numpy``, ``torch``, ``tensorflow``),
each of which have one or more corresponding backends. This way, the generic scattering algorithm can be written in an
architecture-agnostic manner, since all low-level operations are relegated to the backend, and high-level operations
specific to an API are relegated to the frontend.

To make sure that a future pull request (PR) will pass the jenkins and travis tests, please try our package on the
unit tests, the speed as well as the documentation. You might need to install auxiliary libraries via the
``requirements_optional.txt``.

For development purposes, you might need to install the package via::

    git clone https://github.com/kymatio/kymatio.git
    git checkout origin/dev
    cd kymatio
    python setup.py develop

Please refer to `https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow <https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow>`_ for more recommendations.

Backend to frontend: core of the algorithm
==========================================

Common to the 1D, 2D and 3D scattering transform routines are four low-level functions which must be optimized:

1. Fast Fourier transform (FFT) and its inverse (iFFT)
2. Subsampling in the Fourier domain (periodization)
3. Non-linearity (modulus in 1D and 2D, quadratic mean in 3D)
4. Dotwise complex multiplication (``cdgmm``)
5. Padding and unpadding

Checking unit tests
===================

For running all the unit tests and avoiding bugs, please first install the latest versions of ``numpy``, ``tensorflow``,
``torch``, ``cupy``, ``scikit-cuda``. Then, run (in the root directory)::

    pytest

If all the tests pass, you may submit your pull request as explained below. A speed-test is welcome as well.

Checking speed
==============

For checking the speed of a given PR, run the ASV benchmarks on various architectures via::

    cd benchmarks
    run asv

Checking documentation
======================

For checking the documentation, please run the following commands, that will built it through sphinx::

    cd doc
    make clean
    make html

Proposing a pull request
========================

Each PR must be documented using docstrings, illustrated with an example and must pass the unit tests. Please check the
PRs already merged on the GitHub repository if you need an example of a good PR.
