__all__ = ['Scattering2DNumpy']

from ...frontend.numpy_frontend import ScatteringNumpy
from kymatio.scattering2d.core.scattering2d import scattering2d
from ..utils import compute_padding
from ..filter_bank import filter_bank

class Scattering2DNumpy(ScatteringNumpy):
    """ Main module implementing the scattering transform in 2D.
        The scattering transform computes two wavelet transform followed
        by modulus non-linearity.
        It can be summarized as::

            S_J x = [S_J^0 x, S_J^1 x, S_J^2 x]

        for::

            S_J^0 x = x * phi_J
            S_J^1 x = [|x * psi^1_lambda| * phi_J]_lambda
            S_J^2 x = [||x * psi^1_lambda| * psi^2_mu| * phi_J]_{lambda, mu}

        where * denotes the convolution (in space), phi_J is a lowpass
        filter, psi^1_lambda is a family of bandpass
        filters and psi^2_mu is another family of bandpass filters.
        Only Morlet filters are used in this implementation.
        Convolutions are efficiently performed in the Fourier domain.

        Example
        -------
            # 1) Define a Scattering2D object as:
                s = Scattering2D_numpy(J, shape=(M, N))
            #    where (M, N) is the image size and 2**J the scale of the scattering
            # 2) Forward on an input Tensor x of shape B x M x N,
            #     where B is the batch size.
                result_s = s(x)

        Parameters
        ----------
        J : int
            Log-2 of the scattering scale.
        shape : tuple of ints
            Spatial support (M, N) of the input.
        L : int, optional
            Number of angles used for the wavelet transform. Defaults to `8`.
        max_order : int, optional
            The maximum order of scattering coefficients to compute. Must be either
            `1` or `2`. Defaults to `2`.
        pre_pad : boolean, optional
            Controls the padding: if set to False, a symmetric padding is applied
            on the signal. If set to true, the software will assume the signal was
            padded externally. Defaults to `False`.
        backend : object, optional
            Controls the backend which is combined with the frontend.

        Attributes
        ----------
        J : int
            Log-2 of the scattering scale.
        shape : tuple of int
            Spatial support (M, N) of the input.
        L : int, optional
            Number of angles used for the wavelet transform.
        max_order : int, optional
            The maximum order of scattering coefficients to compute.
            Must be either equal to `1` or `2`.
        pre_pad : boolean
            Controls the padding: if set to False, a symmetric padding is applied
            on the signal. If set to true, the software will assume the signal was
            padded externally.
        Psi : dictionary
            Contains the wavelets filters at all resolutions. See
            filter_bank.filter_bank for an exact description.
        Phi : dictionary
            Contains the low-pass filters at all resolutions. See
            filter_bank.filter_bank for an exact description.
        M_padded, N_padded : int
             Spatial support of the padded input.

        Notes
        -----
        The design of the filters is optimized for the value L = 8.

        pre_pad is particularly useful when cropping bigger images because
        this does not introduce border effects inherent to padding.

        """
    def __init__(self, J, shape, L=8, max_order=2, pre_pad=False, backend=None):
        super(Scattering2DNumpy, self).__init__()
        self.pre_pad, self.L, self.backend, self.J, self.shape, self.max_order = pre_pad, L, backend, J, shape,\
                                                                                 max_order
        self.build()

    def build(self):
        self.M, self.N = self.shape
        if not self.backend:
            from ..backend.numpy_backend import backend # is imported like a module and not a class?
            self.backend = backend
        elif self.backend.name[0:5] != 'numpy':
            raise RuntimeError('This backend is not supported.')

        if 2 ** self.J > self.shape[0] or 2 ** self.J > self.shape[1]:
            raise RuntimeError('The smallest dimension should be larger than 2^J')
        self.M_padded, self.N_padded = compute_padding(self.M, self.N, self.J)
        # pads equally on a given side if the amount of padding to add is an even number of pixels, otherwise it adds an extra pixel
        self.pad = self.backend.Pad(
            [(self.M_padded - self.M) // 2, (self.M_padded - self.M + 1) // 2, (self.N_padded - self.N) // 2,
             (self.N_padded - self.N + 1) // 2], [self.M, self.N], pre_pad=self.pre_pad)
        self.unpad = self.backend.unpad
        filters = filter_bank(self.M_padded, self.N_padded, self.J, self.L)
        self.phi, self.psi = filters['phi'], filters['psi']

    def scattering(self, input):
        return scattering2d(input, self.pad, self.unpad, self.backend, self.J, self.L, self.phi, self.psi, self.max_order)

    def loginfo(self):
        return 'NumPy frontend is used.'