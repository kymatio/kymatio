Quick Start
***********

For Linux or MacOS, in bash, do::

    pip install scattering

Then in a python shell::

    import scattering

If the package has been correctly installed, this last command should run.

Specific
========

1-d signals
-----------

2-d signals
-----------

Let us visualize the output after a Scattering Transform::

    import torch
    from scattering import Scattering2D
    scattering = Scattering2D(32, 32, 2)
    x = torch.randn(1, 1, 32, 32)
    Sx = scattering(x)
    print(Sx.size())

This should output::

    torch.Size([1, 1, 81, 8, 8])

Checkout :ref:`user-guide` for more examples of the use of the 2D Scattering Transform

3-d signals
-----------

To go further
=============

Simply check the folder BLAH