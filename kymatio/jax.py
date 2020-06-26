from .scattering1d.frontend.jax_frontend import ScatteringJax1D as Scattering1D

Scattering1D.__module__ = 'kymatio.jax'
Scattering1D.__name__ = 'Scattering1D'


Scattering2D.__module__ = 'kymatio.jax'
Scattering2D.__name__ = 'Scattering2D'

__all__ = ['Scattering1D', 'Scattering2D']
