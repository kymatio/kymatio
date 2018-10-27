"""
Authors: Eugene Belilovsky, Edouard Oyallon and Sergey Zagoruyko
All rights reserved, 2017.
"""

__all__ = ['Scattering']

import warnings
import torch
from .backend import cdgmm, Modulus, SubsampleFourier, fft, Pad, unpad
from .filter_bank import filter_bank
from .utils import compute_padding


class Scattering2D(object):
    def __init__(self, M, N, J, L=8, pre_pad=False):
        """
            Main module implementing the scattering transform in 2D.
            The scattering transform computes two wavelet transform followed
            by modulus non-linearity.
            It can be summarized as:
            S_J x = [S_J^0 x, S_J^1 x, S_J^2 x]
            where
            S_J^0 x = x * phi_J
            S_J^1 x = [|x * psi^1_lambda| * phi_J]_lambda
            S_J^2 x =
                [||x * psi^1_lambda| * psi^2_mu| * phi_J]_{lambda, mu}
            where * denotes the convolution (in space),
            phi_J is a low pass filter, psi^1_lambda is a family of band pass
            filters and psi^2_mu is another family of band pass filters.
            Only Morlet filters are used in this implementation.
            Convolutions are efficiently performed in the Fourier domain
            with this implementation.

            Example
            -------
            # 1) Define a Scattering object as:
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
            pre_pad : boolean, optional
                controls the padding: if set to False, a symmetric padding is applied
                on the signal. If set to true, the software will assume the signal was
                padded externally.

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
            1. The design of the filters is optimized for the value L = 8
            2. pre_pad is particularly useful when doing crops of a
                bigger image because the padding is then extremely accurate. Defaults
                to False.
        """
        self.M, self.N, self.J, self.L = M, N, J, L
        self.pre_pad = pre_pad
        self.build()

    def build(self):
        self.modulus = Modulus()
        self.pad = Pad(2**self.J, pre_pad = self.pre_pad)
        self.subsample_fourier = SubsampleFourier()
        # Create the filters
        self.M_padded, self.N_padded = compute_padding(self.M, self.N, self.J)
        filters = filter_bank(self.M_padded, self.N_padded, self.J, self.L)
        self.Psi = filters['psi']
        self.Phi = [filters['phi'][j] for j in range(self.J)]


    def _type(self, _type):
        for key, item in enumerate(self.Psi):
            for key2, item2 in self.Psi[key].items():
                if torch.is_tensor(item2):
                    self.Psi[key][key2] = item2.type(_type)
        self.Phi = [v.type(_type) for v in self.Phi]
        self.pad.padding_module.type(str(_type).split('\'')[1])
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

    def forward(self, input):
        """
            Forward pass of the scattering.
            Parameters
            ----------
            x : Tensor
                torch Variable with 3 dimensions (B, C, M, N) where (B, C) are arbitrary.
                B typically is the batch size, whereas C is the number of input channels.
            Returns
            -------
            S : Variable tensor or dictionary.
                scattering of the input x, a 4D tensor (B, C, D, M', N') where D corresponds
                to a new channel dimension and (M', N') are downsampled sizes by a factor 2^J.
        """
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

        subsample_fourier = self.subsample_fourier
        modulus = self.modulus
        pad = self.pad

        S = input.new(input.size(0),
                      input.size(1),
                      1 + self.L*J + self.L*self.L*J*(J - 1) // 2,
                      self.M_padded//(2**J)-2,
                      self.N_padded//(2**J)-2)
        U_r = pad(input)
        U_0_c = fft(U_r, 'C2C')  # We trick here with U_r and U_2_c

        # First low pass filter
        U_1_c = subsample_fourier(cdgmm(U_0_c, phi[0]), k=2**J)

        U_J_r = fft(U_1_c, 'C2R')

        S[..., n, :, :] = unpad(U_J_r)
        n = n + 1

        for n1 in range(len(psi)):
            j1 = psi[n1]['j']
            U_1_c = cdgmm(U_0_c, psi[n1][0])
            if(j1 > 0):
                U_1_c = subsample_fourier(U_1_c, k=2 ** j1)
            U_1_c = fft(U_1_c, 'C2C', inverse=True)
            U_1_c = fft(modulus(U_1_c), 'C2C')

            # Second low pass filter
            U_2_c = subsample_fourier(cdgmm(U_1_c, phi[j1]), k=2**(J-j1))
            U_J_r = fft(U_2_c, 'C2R')
            S[..., n, :, :] = unpad(U_J_r)
            n = n + 1

            for n2 in range(len(psi)):
                j2 = psi[n2]['j']
                if(j1 < j2):
                    U_2_c = subsample_fourier(cdgmm(U_1_c, psi[n2][j1]), k=2 ** (j2-j1))
                    U_2_c = fft(U_2_c, 'C2C', inverse=True)
                    U_2_c = fft(modulus(U_2_c), 'C2C')

                    # Third low pass filter
                    U_2_c = subsample_fourier(cdgmm(U_2_c, phi[j2]), k=2 ** (J-j2))
                    U_J_r = fft(U_2_c, 'C2R')

                    S[..., n, :, :] = unpad(U_J_r)
                    n = n + 1

        return S

    def __call__(self, input):
        return self.forward(input)
