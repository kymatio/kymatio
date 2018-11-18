Quick Start
===========

For Linux or MacOS, in bash, do::

    pip install kymatio


Then in a python shell::

    import kymatio

If the package has been correctly installed, this last command should run.


Apply 2D scattering to a 32x32 random image
-------------------------------------------

The following code imports ``torch`` and the 2D Scattering Transform. Then it
creates a ``Scattering2D`` object and applies it to a 32x32 image of random
normal values::

    import torch
    from scattering import Scattering2D
    scattering = Scattering2D(32, 32, 2)
    x = torch.randn(1, 1, 32, 32)
    Sx = scattering(x)
    print(Sx.size())

This should output::

    torch.Size([1, 1, 81, 8, 8])

This corresponds to 81 scattering transform channels with an 8x8 spatial image
each.

Checkout :ref:`user-guide` for more examples of the use of the 2D Scattering Transform

Also make sure you check out the :ref:`backend-story` section to learn how to switch backends


