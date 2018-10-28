Information for developers
**************************

Something something how the code works

Backend: core of the algorithm
==============================

Common to the 1-2-3-D routines of the Scattering Transform, four low-level functions
must be optimized:

1. FFT/iFFT
2. SubsamplingFourier
3. Modulus
4. Dotwise complex multiplication

Unit tests
==========

For running all the unit tests and avoiding bugs, please simply run from the
main folder::

    for each folder in scatteringfolder
        for each file in folder
            pytest x/file.py

If all the tests pass, then you might be able to submit your Pull Request as explained
in the next section!

Proposing a Pull Request(PR)
============================