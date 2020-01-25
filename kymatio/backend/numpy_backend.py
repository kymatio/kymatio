import numpy as np

def modulus(x):
    """
        This function implements a modulus transform for complex numbers.

        Usage
        -----
        x_mod = modulus(x)

        Parameters
        ---------
        x: input complex tensor.

        Returns
        -------
        output: a real tensor equal to the modulus of x.

    """
    return np.abs(x)

def _iscomplex(x):
    return x.dtype == np.complex64 or x.dtype == np.complex128


def _isreal(x):
    return x.dtype == np.float32 or x.dtype == np.float64
