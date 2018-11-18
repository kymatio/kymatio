.. _dev-guide:

Information for developers
**************************

Something something how the code works

Backend: core of the algorithm
==============================

Common to the 1-2-3-D routines of the Scattering Transform, four low-level functions
must be optimized:

1. FFT/iFFT
2. SubsamplingFourier
3. Non-linearity (e.g. modulus for 1-2D)
4. Dotwise complex multiplication
5. Padding/unpadding

Unit tests
==========

For running all the unit tests and avoiding bugs, please simply run from the
main folder::

    pytest

If all the tests pass, then you might be able to submit your Pull Request as explained
in the next section!

Checking speed
==============

Please check `examples/.d/compute_speed.py` to benchmark your modification of backend.

Proposing a Pull Request(PR)
============================

Each PR must be documented using docstings, illustrated with an example and must run the
unit tests.
