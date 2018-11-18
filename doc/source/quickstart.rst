Quick Start
===========

On Linux or macOS, open a shell and run::

    pip install kymatio

In the Python intepreter, you may then call::

    import kymatio

which should run without error if the package has been correctly installed.


Apply 2D scattering to a 32x32 random image
-------------------------------------------

The following code imports ``torch`` and the ``Scattering2D`` class, which
implements the 2D scattering transform. It then creates an instance of this
class to compute the scattering transform at scale ``J = 2`` of a 32x32 image
consisting of Gaussian white noise::

    import torch
    from kymatio import Scattering2D
    scattering = Scattering2D(32, 32, 2)
    x = torch.randn(1, 1, 32, 32)
    Sx = scattering(x)
    print(Sx.size())

This should output::

    torch.Size([1, 1, 81, 8, 8])

This corresponds to 81 scattering coefficients, each corresponding to an
8x8 image.

Check out the :ref:`user-guide` for more scattering transform examples.

The performance of the scattering transform depends on the specific backend
that's in use. For more information on switching backends to improve
performance, see the :ref:`backend-story` section.
