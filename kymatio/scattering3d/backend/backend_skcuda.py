# The `skcuda` backend for Scattering3D is currently under development.
# See issue #179.

NAME = 'skcuda'


skcuda_notimplementederror =\
"""The \"skcuda\" backend for Scattering3D is currently under development.
To perform 3D scattering, please set KYMATIO_BACKEND_3D=\"torch\"."""

def to_complex(input):
    raise NotImplementedError(skcuda_notimplementederror)


def complex_modulus(input_array):
    raise NotImplementedError(skcuda_notimplementederror)


def fft(input, inverse=False):
    """
        fft of a 3d signal

        Example
        -------

        Parameters
        ----------
        input : tensor
            complex input for the FFT
        inverse : bool
            True for computing the inverse FFT.
    """
    raise NotImplementedError(skcuda_notimplementederror)


def cdgmm3d(A, B):
    """
    Pointwise multiplication of complex tensors.

    ----------
    A: complex tensor
    B: complex tensor of the same size as A

    Returns
    -------
    output : tensor of the same size as A containing the result of the
             elementwise complex multiplication of  A with B
    """
    raise NotImplementedError(skcuda_notimplementederror)
