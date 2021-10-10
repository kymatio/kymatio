from .scattering1d.frontend.numpy_frontend import (
    ScatteringNumPy1D as Scattering1D,
    TimeFrequencyScatteringNumPy1D as TimeFrequencyScattering1D)
from .scattering2d.frontend.numpy_frontend import ScatteringNumPy2D as Scattering2D
from .scattering3d.frontend.numpy_frontend \
        import HarmonicScatteringNumPy3D as HarmonicScattering3D

Scattering1D.__module__ = 'kymatio.numpy'
Scattering1D.__name__ = 'Scattering1D'

Scattering2D.__module__ = 'kymatio.numpy'
Scattering2D.__name__ = 'Scattering2D'

HarmonicScattering3D.__module__ = 'kymatio.numpy'
HarmonicScattering3D.__name__ = 'HarmonicScattering3D'

TimeFrequencyScattering1D.__module__ = 'kymatio.numpy'
TimeFrequencyScattering1D.__name__ = 'TimeFrequencyScattering1D'

__all__ = ['Scattering1D', 'Scattering2D', 'HarmonicScattering3D',
           'TimeFrequencyScattering1D']
