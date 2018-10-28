Quick Start
***********

For Linux or MacOS, in bash, do::

    pip install scattering

Then in a python shell::

    import scattering

If the package has been correctly installed, this last command should run.

1-d signals
===========

2-d signals
===========

Let us visualize the output after a Scattering Transform::

    python import torch
    from scattering import Scattering2D
    scattering = Scattering2D(32, 32, 2)
    signal = torch.randn(1, 1, 32, 32)
    output = scattering(signal)
    print(output.size())

3-d signals
===========

To go further
=============

Simply check the folder BLAH