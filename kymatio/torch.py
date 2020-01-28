from .scattering1d.frontend.torch_frontend import ScatteringTorch1D as Scattering1D
from .scattering2d.frontend.torch_frontend import ScatteringTorch2D as Scattering2D
from .scattering3d.frontend.torch_frontend \
        import HarmonicScatteringTorch3D as HarmonicScattering3D

__all__ = ['Scattering1D', 'Scattering2D', 'HarmonicScattering3D']
