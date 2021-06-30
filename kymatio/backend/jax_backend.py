from jax import numpy


from .numpy_backend import NumpyBackend


class JaxBackend(NumpyBackend):
    _np = numpy
    _fft = numpy.fft

    name = 'jax'


