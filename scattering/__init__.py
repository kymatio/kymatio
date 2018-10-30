__all__ = [
            'Scattering3D',
            'Scattering2D',
            'Scattering1D'
            ]

from .scattering2d.scattering2d import Scattering2D
from .scattering1d.scattering1d import Scattering1D
from .scattering3d.scattering3d import Scattering3D

# Make sure that DeprecationWarning within this package always gets printed
### Snippet copied from sklearn.__init__
import warnings
import re
warnings.filterwarnings('always', category=DeprecationWarning,
                        module=r'^{0}.*'.format(re.escape(__name__)))
### End Snippet


