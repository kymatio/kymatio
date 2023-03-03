from .scattering1d.frontend.jax_frontend import ScatteringJax1D as Scattering1D, TimeFrequencyScatteringJax as TimeFrequencyScattering
from .scattering2d.frontend.jax_frontend import ScatteringJax2D as Scattering2D
from .scattering3d.frontend.jax_frontend import (
    HarmonicScatteringJax3D as HarmonicScattering3D,
)

Scattering1D.__module__ = "kymatio.jax"
Scattering1D.__name__ = "Scattering1D"

Scattering2D.__module__ = "kymatio.jax"
Scattering2D.__name__ = "Scattering2D"

HarmonicScattering3D.__module__ = "kymatio.jax"
HarmonicScattering3D.__name__ = "HarmonicScattering3D"


TimeFrequencyScattering.__module__ = "kymatio.jax"
TimeFrequencyScattering.__name__ = "TimeFrequencyScattering"

__all__ = ["Scattering1D", "Scattering2D", "HarmonicScattering3D", "TimeFrequencyScattering"]
