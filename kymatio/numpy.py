from .scattering1d.frontend.numpy_frontend import ScatteringNumPy1D as Scattering1D
from .scattering2d.frontend.numpy_frontend import ScatteringNumPy2D as Scattering2D
from .scattering3d.frontend.numpy_frontend \
        import HarmonicScatteringNumPy3D as HarmonicScattering3D

__all__ = ['Scattering1D', 'Scattering2D', 'HarmonicScattering3D']
