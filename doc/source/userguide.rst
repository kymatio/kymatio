User Guide
==========

Scattering transforms build invariant, stable and informative representations through a non-linear, unitary transform,
 which delocalizes signal information into scattering decomposition paths. They are computed with a cascade of wavelet
 modulus operators, and correspond to a convolutional network where filter coefficients are given by a wavelet operator.
  It is  designed for building representations that incorporate invariants w.r.t. geometrical transformations such as
  translations, rotations or more generally Euclidean transformations.

In practice, Scattering coefficients extend MFSCs or SIFT descriptors, by incorporate second order coefficients that are
 stable but more discriminative than their first order counter part.