.. _user-guide:

User Guide
**********


Introduction to scattering transform
====================================

A Scattering Transform is a non-linear operator that builds
invariant with respect to euclidean geometric transformations such as translations
rotations or change of scale.

A Scattering Network is a complex valued Convolutional Neural Network using predefined
wavelets filters and the complex modulus as non-linearity.  A wavelet transform
propagates a signal throuh each layer, in order to separate structures at different
scales.  Each operator ofthe cascade is non expansive, and so is the cascade.  Thus
higher order layers do not sufferfrom high variance.  In other words, this
representation has mathematical foundations that makes  it  amenable  as a  mainstream
representation  on  structured  signals  such  as  natural images, textures, audio
sounds or molecules.

Let us consider a set of wavelets :math:`\{\psi_\lambda\}_\lambda` adjusted such that
there exists :math:`\epsilon_0` satisfying:

.. math:: 1-\epsilon_0 \leq \sum_\lambda |\hat \psi_\lambda(\omega)|^2 \leq 1

For a fixed :math:`k` and a family of :math:`(\lambda_1,...,\lambda_k)`, a Scattering
coefficient :math:`S[\lambda_1,...,\lambda_k]` of order :math:`k` is such that:

.. math:: S[\lambda_1,...,\lambda_k] = |\psi_{\lambda_k} \star ...| \psi_{\lambda_1} \star x|...|

See :cite:`mallat2012group` for deeper mathematical insights on the Scattering Transform.

Practical implementation
========================

Former implementation of the Scattering Transform computed each Scattering coefficients
layer per layer. Here, we visit the tree of the Scattering path node per node. It permits
to limit the memory use, and thus benefiting from CUDA accelerations. This is explained on
REFERENE_FIGURE.

.. image:: _static/algorithm.png
   :width: 600px
   :alt: Graph of the algorithm used
   :align: center

More details about our implementation can be found in SECTION_DU_DEV


1-D
---

2-D
---

The 2D implementation in this package  provided  outputs coefficients which
matches exactly the one described in :cite:`anden2014scatnet`.


3-D
---

Examples
========

1-D
---

2-D
---

3-D
---

Output size
===========

1-D
---


2-D
---

Let us assume that :math:`x` is a tensor of size :math:`(B,C,N_1,N_2)`. Then, if the
output :math:`Sx` via a Scattering Transform with scale :math:`J` and :math:`L` angles will have
size:


.. math:: (B,C,1+LJ+\frac{L^2J(J-1)}{2},\frac{N_1}{2^J},\frac{N_2}{2^J})

3-D
---

Switching devices: cuda>cpu or cuda<cpu
=======================================

By default, the Scattering Transform is run on CPU::

    import torch
    from scattering import Scattering2D
    scattering = Scattering2D(32, 32, 2)
    x = torch.randn(1, 1, 32, 32)
    Sx = scattering(x)

However, if a GPU combined with CUDA is available, then it is possible to run it on GPU via::

    scattering.cuda()
    x = x.cuda()
    Sx_ = scattering(x)
    print(torch.norm(Sx_-Sx)

Then, it is possible to redo the computations on CPU via::

    scattering.cpu()
    x = x.cpu()
    Sx = scattering(x)
    print(torch.norm(Sx_-Sx)

.. _backend-story:

Backend
=======

This package is maintained with a flexible backend that currently supports PyTorch. A
backend corresponds to an implementation of routines, which are optimized for their
final purpose. For instance, `torch` backend is slightly slower than others backend
but it has the advantage to be differentiable.

At installation time, a config files is created in `~/.config/scattering/config.cfg` that
will contain a backend used by default. This default backend will be overwritten if
a global environment variable `SCATTERING_BACKEND` is created and not equal to `None`
and in this case, each backends will use `SCATTERING_BACKEND` as a default backend.
It is possible to specify more precisely the backend that will be used for each
signal type as we will see below.

1-D backend
-----------


2-D backend
-----------

If the global environment variable `SCATTERING_BACKEND_2D` is not equal to `None`, then
its value will be used at running time as the backend. Currently, two backends exist:

- `torch`: the scattering is differentiable w.r.t. its parameters, however it can be too slow to be amenable for large scale classification.

- `skcuda`: the scattering is not differentiable but is optimized to deliver fast computations.

3-D backend
-----------

Benchmark with previous versions
================================

1-D backend
-----------

2-D backend
-----------

We compared our implementation with MATLAB version :cite:`Oyallon_2015_CVPR` and *PyScatWave* former
PyTorch implementation :cite:`8413168`. The following table correspond to the average compute time
for a batch of size :math:`128\times 3 \times 256 \times 256`, for a forward pass:

====================================            =========================
Name                                            Average time(s) per batch
====================================            =========================
MATLAB :cite:`Oyallon_2015_CVPR`                >200
Kymatio (torch backend, CPU)                    110
Kymatio (torch backend, 1080Ti GPU)             4.4
Kymatio (torch backend, V100 GPU)               2.9
PyScatWave :cite:`8413168`                      1.6
Kymatio (skcuda backend, 1080Ti GPU)            1.1
Kymatio (skcuda backend, V100 GPU)              0.49
====================================            =========================

The CPU tests were performed on a 48-core machine.


3-D backend
-----------


.. rubric:: References

.. bibliography:: _static/bibtex.bib
