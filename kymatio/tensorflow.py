from .scattering1d.frontend.tensorflow_frontend import ScatteringTensorFlow1D as Scattering1D
from .scattering2d.frontend.tensorflow_frontend import ScatteringTensorFlow2D as Scattering2D
from .scattering3d.frontend.tensorflow_frontend \
        import HarmonicScatteringTensorFlow3D as HarmonicScattering3D

Scattering1D.__module__ = 'kymatio.tensorflow'
Scattering1D.__name__ = 'Scattering1D'

Scattering2D.__module__ = 'kymatio.tensorflow'
Scattering2D.__name__ = 'Scattering2D'

HarmonicScattering3D.__module__ = 'kymatio.tensorflow'
HarmonicScattering3D.__name__ = 'HarmonicScattering3D'

__all__ = ['Scattering1D', 'Scattering2D', 'HarmonicScattering3D']
