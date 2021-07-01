from jax import numpy


from .numpy_backend import NumpyBackend


class JaxBackend(NumpyBackend):
    _np = numpy
    _fft = numpy.fft

    name = 'jax'

    @staticmethod
    def input_checks(x):
        if x is None:
            raise TypeError('The input should be not empty.')

