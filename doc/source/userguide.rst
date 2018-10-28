User Guide
**********

Introduction to scattering transform
====================================

A scattering transform is a non-linear operator that builds
invariant with respect to euclidean geometric transformations such as translations
rotations or change of scale.

A Scattering Network is a complex valued Convolutional Neural Network using predefined
wavelets filters and the complex modulus as nonlinearity.  A wavelet transform
propagates asignal throuh each layer, in order to separate structures at different
scales.  Each operator ofthe cascade is non expansive, and so is the cascade.  Thus
higher order layers do not sufferfrom high variance.  In other words, this
representation has mathematical foundations thatmakes  it  amenable  for  mainstream
statistical  use  on  structured  signals  such  as  naturalimages, textures, audio
sounds or mollecules.

Let us consider a set of wavelets :math:`\{\psi_\lambda\}_\lambda` adjusted such that
there exists :math:`\epsilon_0` satisfying:

.. math:: 1-\epsilon_0 \leq \sum_\lambda |\hat \psi_\lambda(\omega)|^2 \leq 1

Output size
===========

Switching devices: cuda>cpu or cuda<cpu
=======================================

Backend
=======

1-D backend
-----------

A backend system is implemented, using.

2-D backend
-----------


3-D backend
-----------