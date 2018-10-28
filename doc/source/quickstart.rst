Quick Start
***********

For Linux or MacOS, in bash, do::

    pip install scattering

Then in a python shell::

    import scattering

If the package has been correctly installed, this last command should run.

Or for 2D signals::

    python import torch
    from scattering import Scattering2D
    scattering = Scattering2D(32, 32, 2)
    signal = torch.randn(1, 1, 32, 32)
    output = scattering(signal)


