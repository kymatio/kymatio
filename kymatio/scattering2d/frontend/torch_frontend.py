__all__ = ['Scattering2DTorch']

import torch
import torch.nn as nn

from kymatio.scattering2d.core.scattering2d import scattering2d
from ..filter_bank import filter_bank
from ..utils import compute_padding
from ...frontend.torch_frontend import ScatteringTorch


class Scattering2DTorch(ScatteringTorch):
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
                s = Scattering2D_torch(J, shape=(M, N))
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
        super(Scattering2DTorch, self).__init__()
        self.pre_pad, self.L, self.backend, self.J, self.shape, self.max_order = pre_pad, L, backend, J, shape,\
                                                                                 max_order
        self.build()

    def build(self):
        self.M, self.N = self.shape
        # use the default backend if no backend is provided
        if not self.backend:
            from ..backend.torch_backend import backend
            self.backend = backend
        elif self.backend.name[0:5] != 'torch':
            raise RuntimeError('This backend is not supported.')

        if 2 ** self.J > self.shape[0] or 2 ** self.J > self.shape[1]:
            raise RuntimeError('The smallest dimension should be larger than 2^J.')
        self.M_padded, self.N_padded = compute_padding(self.M, self.N, self.J)
        # pads equally on a given side if the amount of padding to add is an even number of pixels, otherwise it adds an extra pixel
        self.pad = self.backend.Pad([(self.M_padded - self.M) // 2, (self.M_padded - self.M+1) // 2, (self.N_padded - self.N) // 2,
                                (self.N_padded - self.N + 1) // 2], [self.M, self.N], pre_pad=self.pre_pad)
        self.unpad = self.backend.unpad
        self.create_and_register_filters()

    def create_and_register_filters(self):
        """ This function run the filterbank function that
            will create the filters as numpy array, and then, it
            saves those arrays as module's buffers."""

        # Create the filters
        filters = filter_bank(self.M_padded, self.N_padded, self.J, self.L)
        n = 0
        self.phi, self.psi = filters['phi'], filters['psi']
        for c, phi in self.phi.items():
            if isinstance(c, int):
                self.phi[c] = torch.from_numpy(self.phi[c]).unsqueeze(-1) # add a trailing singleton dimension to mark
                # it as non-complex
                self.register_buffer('tensor' + str(n), self.phi[c])
                n += 1

        for j in range(len(self.psi)):
            for k, v in self.psi[j].items():
                if isinstance(k, int):
                    self.psi[j][k] = torch.from_numpy(v).unsqueeze(-1) # add a trailing singleton dimension to mark it
                    # as non-complex
                    self.register_buffer('tensor' + str(n), self.psi[j][k])
                    n += 1

    def scattering(self, input):
        # each time scattering is run, one needs to make sure self.psi and self.phi point to
        # the correct buffers
        n = 0
        buffer_dict = dict(self.named_buffers())
        for c, phi in self.phi.items():
            if isinstance(c, int):
                self.phi[c] =  buffer_dict['tensor' + str(n)]
                n += 1

        for j in range(len(self.psi)):
            for k, v in self.psi[j].items():
                if isinstance(k, int):
                    self.psi[j][k] = buffer_dict['tensor' + str(n)]
                    n += 1

        return scattering2d(input, self.pad, self.unpad, self.backend, self.J, self.L, self.phi, self.psi, self.max_order)

    def forward(self, input):
        """Forward pass of the scattering.

        Parameters
        ----------
        input : tensor
           Tensor with k+2 dimensions :math:`(n_1, ..., n_k, M, N)` where :math:`(n_1, ...,n_k)` is
           arbitrary. Currently, k=2 is hardcoded. :math:`n_1` typically is the batch size, whereas
            :math:`n_2` is the number of
           input channels.

        Returns
        -------
        S : tensor
           Scattering of the input, a tensor with k+3 dimensions :math:`(n_1, ...,n_k, D, Md, Nd)`
           where :math:`D` corresponds to a new channel dimension and :math:`(Md, Nd)` are
           downsampled sizes by a factor :math:`2^J`. Currently, k=2 is hardcoded.

        """
        if not torch.is_tensor(input):
            raise TypeError('The input should be a PyTorch Tensor.')

        if len(input.shape) < 2:
            raise RuntimeError('Input tensor must have at least two dimensions.')

        if not input.is_contiguous():
            raise RuntimeError('Tensor must be contiguous!')

        if (input.size(-1) != self.N or input.size(-2) != self.M) and not self.pre_pad:
            raise RuntimeError('Tensor must be of spatial size (%i,%i)!' % (self.M, self.N))

        if (input.size(-1) != self.N_padded or input.size(-2) != self.M_padded) and self.pre_pad:
            raise RuntimeError('Padded tensor must be of spatial size (%i,%i)!' % (self.M_padded, self.N_padded))

        return self.scattering(input)

    def loginfo(self):
        return 'Torch frontend is used.'