from ...backend.jax_backend import JaxBackend
from .numpy_backend import NumpyBackend1D


class JaxBackend1D(JaxBackend, NumpyBackend1D):
    # JaxBackend1D inherits _np and _fft and name attributes from JaxBackend
    # The rest comes from NumpyBackend1D
    # and from NumpyBackend via both JaxBackend and NumpyBackend1D

    # The modification of _np and _fft are sufficient to 
    # drop in jax.numpy instead of numpy. This works because
    # all the numpy commands have an equivalent in jax.numpy

    pass

backend = JaxBackend1D
