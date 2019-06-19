__all__ = ['Scattering2D']

import torch
from ..backend import cdgmm, Modulus, SubsampleFourier, fft, Pad, unpad, NAME
from ..filter_bank import filter_bank
from ..utils import compute_padding
from ..scattering2d import scattering2d
from ...frontend.torch_f import Scattering

if NAME != 'torch' and NAME != 'skcuda':
    raise (RuntimeError('The only supported backend by the torch frontend are torch and skcuda.'))

class Scattering2D(Scattering):
    """Main module implementing the scattering transform in 2D.
        The scattering transform computes two wavelet transform followed
        by modulus non-linearity.
        It can be summarized as::

            S_J x = [S_J^0 x, S_J^1 x, S_J^2 x]

        where::

            S_J^0 x = x * phi_J
            S_J^1 x = [|x * psi^1_lambda| * phi_J]_lambda
            S_J^2 x = [||x * psi^1_lambda| * psi^2_mu| * phi_J]_{lambda, mu}

        where * denotes the convolution (in space), phi_J is a low pass
        filter, psi^1_lambda is a family of band pass
        filters and psi^2_mu is another family of band pass filters.
        Only Morlet filters are used in this implementation.
        Convolutions are efficiently performed in the Fourier domain
        with this implementation.

        Example
        -------
            # 1) Define a Scattering object as:
            s = Scattering2D(J, shape=(M, N))
            #    where (M, N) are the image sizes and 2**J the scale of the scattering
            # 2) Forward on an input Tensor x of shape B x M x N,
            #     where B is the batch size.
            result_s = s(x)

        Parameters
        ----------
        J : int
            logscale of the scattering
        shape : tuple of int
            spatial support (M, N) of the input
        L : int, optional
            number of angles used for the wavelet transform
        max_order : int, optional
            The maximum order of scattering coefficients to compute. Must be either
            `1` or `2`. Defaults to `2`.
        pre_pad : boolean, optional
            controls the padding: if set to False, a symmetric padding is applied
            on the signal. If set to true, the software will assume the signal was
            padded externally.

        Attributes
        ----------
        J : int
            logscale of the scattering
        shape : tuple of int
            spatial support (M, N) of the input
        L : int, optional
            number of angles used for the wavelet transform
        max_order : int, optional
            The maximum order of scattering coefficients to compute.
            Must be either equal to `1` or `2`. Defaults to `2`.
        pre_pad : boolean
            controls the padding
        Psi : dictionary
            containing the wavelets filters at all resolutions. See
            filter_bank.filter_bank for an exact description.
        Phi : dictionary
            containing the low-pass filters at all resolutions. See
            filter_bank.filter_bank for an exact description.
        M_padded, N_padded : int
             spatial support of the padded input

        Notes
        -----
        The design of the filters is optimized for the value L = 8

        pre_pad is particularly useful when doing crops of a bigger
         image because the padding is then extremely accurate. Defaults
         to False.

        """
    def __init__(self, J, shape, L=8, max_order=2, pre_pad=False):
        super(Scattering2D, self).__init__(J, shape, max_order = max_order)
        self.L, self.J, self.max_order = L, J, max_order
        self.pre_pad = pre_pad
        self.shape = shape
        if 2 ** J > shape[0] or 2 ** J > shape[1]:
            raise (RuntimeError('The smallest dimension should be larger than 2^J'))
        self.build()

    def build(self):
        self.M, self.N = self.shape
        self.modulus = Modulus()
        self.M_padded, self.N_padded = compute_padding(self.M, self.N, self.J)
        # pads equally on a given side if the amount of padding to add is an even number of pixels, otherwise it adds an extra pixel
        self.pad = Pad([(self.N_padded - self.N) // 2, (self.N_padded - self.N + 1) // 2, (self.M_padded - self.M) // 2,
                       (self.M_padded - self.M + 1) // 2], [self.N, self.M], pre_pad=self.pre_pad)
        self.subsample_fourier = SubsampleFourier()
        self.filters = self.create_and_register_filters()

    def create_and_register_filters(self):
        # Create the filters
        filters = filter_bank(self.M_padded, self.N_padded, self.J, self.L)
        n = 0
        self.phi, self.psi = filters['phi'], filters['psi']
        for c, phi in self.phi.items():
            if isinstance(c, int):
                self.phi[c] = torch.from_numpy(self.phi[c])
                self.register_buffer('tensor' + str(n), self.phi[c])
                n += 1

        for j in range(len(self.psi)):
            for k, v in self.psi[j].items():
                if isinstance(k, int):
                    self.psi[j][k] = torch.from_numpy(v)
                    self.register_buffer('tensor' + str(n), self.psi[j][k])
                    n += 1

    def scattering(self, input):
        # each time scattering is run, one needs to make sure self.psi and self.phi point to
        # the correct buffers
        n=0
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

        return scattering2d(input, self.J, self.L, self.subsample_fourier, self.pad, self.modulus, fft, cdgmm, unpad, self.phi, self.psi, self.max_order, self.M_padded, self.N_padded)

    def forward(self, input):
        """Forward pass of the scattering.

        Parameters
        ----------
        input : tensor
           tensor with 3 dimensions :math:`(B, C, M, N)` where :math:`(B, C)` are arbitrary.
           :math:`B` typically is the batch size, whereas :math:`C` is the number of input channels.

        Returns
        -------
        S : tensor
           scattering of the input, a 4D tensor :math:`(B, C, D, Md, Nd)` where :math:`D` corresponds
           to a new channel dimension and :math:`(Md, Nd)` are downsampled sizes by a factor :math:`2^J`.

        """
        if not torch.is_tensor(input):
            raise (
            TypeError('The input should be a torch.cuda.FloatTensor, a torch.FloatTensor or a torch.DoubleTensor'))

        if len(input.shape) < 2:
            raise (RuntimeError('Input tensor must have at least two '
                               'dimensions'))

        if (not input.is_contiguous()):
            raise (RuntimeError('Tensor must be contiguous!'))

        if ((input.size(-1) != self.N or input.size(-2) != self.M) and not self.pre_pad):
            raise (RuntimeError('Tensor must be of spatial size (%i,%i)!' % (self.M, self.N)))

        if ((input.size(-1) != self.N_padded or input.size(-2) != self.M_padded) and self.pre_pad):
            raise (RuntimeError('Padded tensor must be of spatial size (%i,%i)!' % (self.M_padded, self.N_padded)))

        return self.scattering(input)