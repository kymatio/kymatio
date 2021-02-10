from .scattering1d.frontend.sklearn_frontend \
    import ScatteringTransformer1D as Scattering1D
from .scattering2d.frontend.sklearn_frontend \
    import ScatteringTransformer2D as Scattering2D
from .scattering3d.frontend.sklearn_frontend \
    import HarmonicScatteringTransformer3D as HarmonicScattering3D

Scattering1D.__module__ = 'kymatio.sklearn'
Scattering1D.__name__ = 'Scattering1D'

Scattering2D.__module__ = 'kymatio.sklearn'
Scattering2D.__name__ = 'Scattering2D'

HarmonicScattering3D.__module__ = 'kymatio.sklearn'
HarmonicScattering3D.__name__ = 'HarmonicScattering3D'

__all__ = ['Scattering1D', 'Scattering2D', 'HarmonicScattering3D']
