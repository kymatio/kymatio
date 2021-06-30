from ...backend.jax_backend import JaxBackend
from .numpy_backend import NumpyBackend1D


class JaxBackend1D(JaxBackend, NumpyBackend1D):
    pass

backend = JaxBackend1D
