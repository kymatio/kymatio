from .scattering1d.frontend.keras_frontend \
    import ScatteringKeras1D as Scattering1D
from .scattering2d.frontend.keras_frontend \
    import ScatteringKeras2D as Scattering2D

Scattering1D.__module__ = 'kymatio.keras'
Scattering1D.__name__ = 'Scattering1D'

Scattering2D.__module__ = 'kymatio.keras'
Scattering2D.__name__ = 'Scattering2D'

__all__ = ['Scattering1D', 'Scattering2D']
