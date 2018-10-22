"""
Authors: Eugene Belilovsky, Edouard Oyallon and Sergey Zagoruyko
All rights reserved, 2017.
"""

__all__ = ['Scattering']

import warnings
import torch
from .utils import cdgmm, Modulus, Periodize, Fft
from .filters_bank import filters_bank
from torch.legacy.nn import SpatialReflectionPadding as pad_function


class Scattering2D(object):
    """
        Main module implementing the scattering transform in 2D.

        The scattering transform computes two wavelet transform followed
        by modulus non-linearity.
        It can be summarized as:
        S_J x = [S_J^0 x, S_J^1 x, S_J^2 x]
        where
        S_J^0 x = x star phi_J
        S_J^1 x = [|x star psi^1_lambda| star phi_J]_lambda
        S_J^2 x =
            [||x star psi^1_lambda| star psi^2_mu| star phi_J]_{lambda, mu}

        where star denotes the convolution (in space),
        phi_J is a low pass filter, psi^1_lambda is a family of band pass
        filters and psi^2_lambda is another family of band pass filters.

        Only Morlet filters are used in this implementation.

        Convolutions are efficiently performed in the Fourier domain
        with this implementation.

        Example
        -------
        # 1) Define a Scatterer object as:
        s = Scattering2D(M, N, J)
        #    where (M, N) are the image sizes and 2**J the scale of the scattering
        # 2) Forward on an input Variable x of shape B x 1 x M x N,
        #     where B is the batch size.
        result_s = s(x)


        Parameters
        ----------
        M, N : int
            spatial support of the input
        J : int
            logscale of the scattering
        L : int, optional
            number of angles used for the wavelet transform
            N.B.: the design of the filters is optimized for the value L = 8
        pre_pad : boolean, optional
            controls the padding: if set to False, a symmetric padding is applied
            on the signal. If set to true, the software will assume the signal was
            padded externally. This is particularly useful when doing crops of a
            bigger image because the padding is then extremely accurate. Defaults
            to False.
        backend : string, optional
            backend used, which affects substantially functionality and
            performances. When backend is set to 'torch', one can access to the
            gradient w.r.t. the inputs, using pure torch routines. However, this
            can be substantially slower than using 'skcuda' backend that uses
            optimized cuda kernels but is not differentiable. Defaults to 'torch'.

        Attributes
        ----------
        M, N : int
            spatial support of the input
        J : int
            logscale of the scattering
        L : int, optional
            number of angles used for the wavelet transform
        pre_pad : boolean
            controls the padding
        backend : string
            backend used
        Psi : dictionary
            countaining the wavelets filters at all resolutions. See
            filters_bank.filters_bank_real for an exact description.
        Psi : dictionary
            countaining the low-pass filters at all resolutions. See
            filters_bank.filters_bank_real for an exact description.
        fft : class
            FFT class
        modulus : class
            complex module class
        periodize : class
            periodize class for fourier signals. It acts as a downsampling
            in the spatial domain.
        padding_module : function
            function used to pad input signals. Currently, it relies on torch.legacy
    """
    def __init__(self, M, N, J, L=8, pre_pad=False, backend='torch'):
        self.M, self.N, self.J, self.L = M, N, J, L
        self.pre_pad = pre_pad
        self.backend = backend
        self.fft = Fft()
        self.modulus = Modulus(backend=backend)
        self.periodize = Periodize(backend=backend)

        self._prepare_padding_size([1, 1, M, N])

        self.padding_module = pad_function(2**J)

        # Create the filters
        filters = filters_bank(self.M_padded, self.N_padded, J, L)

        self.Psi = filters['psi']
        self.Phi = [filters['phi'][j] for j in range(J)]

    def _type(self, _type):
        for key, item in enumerate(self.Psi):
            for key2, item2 in self.Psi[key].items():
                if torch.is_tensor(item2):
                    self.Psi[key][key2] = item2.type(_type)
        self.Phi = [v.type(_type) for v in self.Phi]
        self.padding_module.type(str(_type).split('\'')[1])
        return self

    def cuda(self):
        """
            Moves the parameters of the scattering to the GPU
        """
        return self._type(torch.cuda.FloatTensor)

    def cpu(self):
        """
            Moves the parameters of the scattering to the CPU
        """
        return self._type(torch.FloatTensor)

    def _prepare_padding_size(self, s):
        M = s[-2]
        N = s[-1]

        self.M_padded = ((M + 2 ** (self.J))//2**self.J+1)*2**self.J
        self.N_padded = ((N + 2 ** (self.J))//2**self.J+1)*2**self.J

        if self.pre_pad:
            warnings.warn('Make sure you padded the input before to feed it!', RuntimeWarning, stacklevel=2)

        s[-2] = self.M_padded
        s[-1] = self.N_padded
        self.padded_size_batch = torch.Size([a for a in s])

    # This function copies and view the real to complex
    def _pad(self, input):
        if(self.pre_pad):
            output = input.new(input.size(0), input.size(1), input.size(2), input.size(3), 2).fill_(0)
            output.narrow(output.ndimension()-1, 0, 1).copy_(input)
        else:
            out_ = self.padding_module.updateOutput(input)
            output = input.new(*(out_.size() + (2,))).fill_(0)
            output.select(4, 0).copy_(out_)
        return output

    def _unpad(self, in_):
        return in_[..., 1:-1, 1:-1]

    def forward(self, input):
        if not torch.is_tensor(input):
            raise(TypeError('The input should be a torch.cuda.FloatTensor, a torch.FloatTensor or a torch.DoubleTensor'))

        if (not input.is_contiguous()):
            raise (RuntimeError('Tensor must be contiguous!'))

        if((input.size(-1)!=self.N or input.size(-2)!=self.M) and not self.pre_pad):
            raise (RuntimeError('Tensor must be of spatial size (%i,%i)!'%(self.M,self.N)))

        if ((input.size(-1) != self.N_padded or input.size(-2) != self.M_padded) and self.pre_pad):
            raise (RuntimeError('Padded tensor must be of spatial size (%i,%i)!' % (self.M_padded, self.N_padded)))

        if (input.dim() != 4):
            raise (RuntimeError('Input tensor must be 4D'))

        J = self.J
        phi = self.Phi
        psi = self.Psi
        n = 0

        fft = self.fft
        periodize = self.periodize
        modulus = self.modulus
        pad = self._pad
        unpad = self._unpad

        S = input.new(input.size(0),
                      input.size(1),
                      1 + self.L*J + self.L*self.L*J*(J - 1) // 2,
                      self.M_padded//(2**J)-2,
                      self.N_padded//(2**J)-2)
        U_r = pad(input)
        U_0_c = fft(U_r, 'C2C')  # We trick here with U_r and U_2_c

        # First low pass filter
        U_1_c = periodize(cdgmm(U_0_c, phi[0], backend=self.backend), k=2**J)

        U_J_r = fft(U_1_c, 'C2R')

        S[..., n, :, :] = unpad(U_J_r)
        n = n + 1

        for n1 in range(len(psi)):
            j1 = psi[n1]['j']
            U_1_c = cdgmm(U_0_c, psi[n1][0], backend=self.backend)
            if(j1 > 0):
                U_1_c = periodize(U_1_c, k=2 ** j1)
            U_1_c = fft(U_1_c, 'C2C', inverse=True)
            U_1_c = fft(modulus(U_1_c), 'C2C')

            # Second low pass filter
            U_2_c = periodize(cdgmm(U_1_c, phi[j1], backend=self.backend), k=2**(J-j1))
            U_J_r = fft(U_2_c, 'C2R')
            S[..., n, :, :] = unpad(U_J_r)
            n = n + 1

            for n2 in range(len(psi)):
                j2 = psi[n2]['j']
                if(j1 < j2):
                    U_2_c = periodize(cdgmm(U_1_c, psi[n2][j1], backend=self.backend), k=2 ** (j2-j1))
                    U_2_c = fft(U_2_c, 'C2C', inverse=True)
                    U_2_c = fft(modulus(U_2_c), 'C2C')

                    # Third low pass filter
                    U_2_c = periodize(cdgmm(U_2_c, phi[j2], backend=self.backend), k=2 ** (J-j2))
                    U_J_r = fft(U_2_c, 'C2R')

                    S[..., n, :, :] = unpad(U_J_r)
                    n = n + 1

        return S

    def __call__(self, input):
        return self.forward(input)
