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

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< e50eb03a008e6a4da762782cb4de667bdfae117f
from .scattering1d import Scattering1D
from .scattering2d import Scattering2D
from .scattering3d import HarmonicScattering3D
=======
from .scattering1d.scattering1d import Scattering1D
from .scattering2d.scattering2d import Scattering2D
from .scattering3d.scattering3d import HarmonicScattering3D
>>>>>>> Revert "adding all basic files"
=======
from .scattering1d import Scattering1D
from .scattering2d import Scattering2D

from .scattering3d import HarmonicScattering3D
>>>>>>> e801b80... adding all basic files
=======
from .scattering1d.scattering1d import Scattering1D
from .scattering2d.scattering2d import Scattering2D
from .scattering3d.scattering3d import HarmonicScattering3D
>>>>>>> fbdea27... Revert "adding all basic files"
=======
from .scattering1d import Scattering1D
from .scattering2d import Scattering2D
from .scattering3d import HarmonicScattering3D
>>>>>>> aef5ab7... forgot a thing

from .version import version as __version__

