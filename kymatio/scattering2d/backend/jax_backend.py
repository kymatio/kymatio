from ...backend.jax_backend import JaxBackend
from .numpy_backend import NumpyBackend2D


class JaxBackend2D(JaxBackend, NumpyBackend2D):
    # JaxBackend2D inherits _np and _fft and name attributes from JaxBackend
    # The rest comes from NumpyBackend2D
    # and from NumpyBackend via both JaxBackend and NumpyBackend2D

    # The modification of _np and _fft are sufficient to 
    # drop in jax.numpy instead of numpy. This works because
    # all the numpy commands have an equivalent in jax.numpy

    pass

backend = JaxBackend2D
