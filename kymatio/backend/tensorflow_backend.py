import numpy as np
import tensorflow as tf

def _iscomplex(x):
    return x.dtype == np.complex64 or x.dtype == np.complex128

def _isreal(x):
    return x.dtype == np.float32 or x.dtype == np.float64

class Modulus(object):
    """This class implements a modulus transform for complex numbers.

    Parameters
    ----------
    x: input complex tensor.

    Returns
    ----------
    output: a real tensor equal to the modulus of x.

    Usage
    ----------
    modulus = Modulus()
    x_mod = modulus(x)
    """
    def __call__(self, x):
        norm = tf.abs(x)
        return tf.cast(norm, tf.complex64)
