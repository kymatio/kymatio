# Make sure that DeprecationWarning within this package always gets printed
### Snippet copied from sklearn.__init__
import warnings
import re

warnings.filterwarnings('always', category=DeprecationWarning,
                        module=r'^{0}.*'.format(re.escape(__name__)))
warnings.filterwarnings('always', category=PendingDeprecationWarning,
                        module=r'^{0}.*'.format(re.escape(__name__)))
### End Snippet

__all__ = [
            'Scattering1D',
            'TimeFrequencyScattering',
            'Scattering2D',
            'HarmonicScattering3D'
            ]

from .scattering1d import ScatteringEntry1D as Scattering1D
from .scattering1d import TimeFrequencyScatteringEntry as TimeFrequencyScattering
from .scattering2d import ScatteringEntry2D as Scattering2D
from .scattering3d import HarmonicScatteringEntry3D as HarmonicScattering3D

from .version import version as __version__
