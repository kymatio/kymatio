__all__ = [
            'Scattering2D',
            'Scattering',
            'Scattering1D'
            ]

from .scattering2d.scattering2d import Scattering2D
from .scattering1d import Scattering1D

from .datasets import fetch_fsdd
from .caching import get_cache_dir



# Make sure that DeprecationWarning within this package always gets printed
### Snippet copied from sklearn.__init__
import warnings
import re
warnings.filterwarnings('always', category=DeprecationWarning,
                        module=r'^{0}.*'.format(re.escape(__name__)))
### End Snippet

class Scattering(Scattering2D):
    def __init__(self, M, N, J, pre_pad=False, jit=True):
        warnings.warn("Scattering is deprecated in release 0.2. "
                      "Please use Scattering2D instead.",
                      category=DeprecationWarning)
        super(Scattering, self).__init__(M, N, J, pre_pad, jit)


