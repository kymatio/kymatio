.. _dev-guide:

Information for developers
**************************

Kymatio implements the scattering transform for different frontends (currently ``numpy``, ``torch``, ``tensorflow``), each of
which have one or more corresponding backends. This way, the generic scattering algorithm can be written in an architecture-agnostic manner,
since all low-level operations are relegated to the backend, and high-level operations specific to an API are relegated
to the frontend.

Backend: core of the algorithm
==============================

Common to the 1D, 2D and 3D scattering transform routine are four low-level functions which must be optimized:

1. Fast Fourier transform (FFT) and its inverse (IFFT)
2. Subsampling in the Fourier domain (periodization)
3. Non-linearity (modulus in 1D and 2D, quadratic mean in 3D)
4. Dotwise complex multiplication (``cdgmm``)
5. Padding and unpadding

Unit tests
==========

For running all the unit tests and avoiding bugs, please first install the latest versions of ``numpy``, ``tensorflow``,
``torch``, ``cupy``, ``scikit-cuda`` .Then, run (in the root directory)::

    pytest

If all the tests pass, you may submit your pull request as explained below.

Checking speed
==============

Please check out the `examples/*d/compute_speed.py` scripts to benchmark your modifications.

Proposing a pull request
========================

Each pull request (PR) must be documented using docstrings, illustrated with an example and must pass the unit tests.
