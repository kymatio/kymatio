.. _dev-guide:

Information for developers
**************************

Kymatio implements the scattering transform for different architectures (currently ``torch`` and ``scikit-cuda``/``cupy``) through backends. This way, the generic algorithm can be written in an architecture-agnostic manner, since all low-level operations are relegated to the backend.

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

For running all the unit tests and avoiding bugs, please run (in the root directory)::

    pytest

If all the tests pass, you may submit your pull request as explained below.

Checking speed
==============

Please check out the `examples/*d/compute_speed.py` scripts to benchmark your modifications.

Proposing a pull request
========================

Each pull request (PR) must be documented using docstrings, illustrated with an example and must pass the unit tests.
