# Make sure that DeprecationWarning within this package always gets printed
### Snippet copied from sklearn.__init__
import warnings
import re
warnings.filterwarnings('always', category=DeprecationWarning,
                        module=r'^{0}.*'.format(re.escape(__name__)))
### End Snippet

__all__ = [
            'Scattering1D',
            'Scattering2D',
            'HarmonicScattering3D'
            ]

from .scattering1d.scattering1d import Scattering1D
from .scattering2d.scattering2d import Scattering2D
from .scattering3d.scattering3d import HarmonicScattering3D

from .version import version as __version__
