"""
Authors: Eugene Belilovsky, Edouard Oyallon and Sergey Zagoruyko
All rights reserved, 2017.
"""

__all__ = ['Scattering']

import warnings
import torch
from .backend import cdgmm, Modulus, SubsampleFourier, fft
from .filters_bank import filters_bank
from .utils import prepare_padding_size


class Scattering2D(object):
    """Scattering module.

    Runs scattering on an input image in NCHW format

    Input args:
        M, N: input image size
        J: number of layers
        pre_pad: if set to True, module expect pre-padded images
        jit: compile kernels on the fly for speed
    """
    def __init__(self, M, N, J, L=8, pre_pad=False, backend='torch'):
        self.M, self.N, self.J, self.L = M, N, J, L
        self.pre_pad = pre_pad
        self.backend = backend
        self.modulus = Modulus()
        self.subsample_fourier = SubsampleFourier()
        self.build()

    def build(self):
        # Create the filters
        self.M_padded, self.N_padded = prepare_padding_size(self.M, self.N)
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
        return self._type(torch.cuda.FloatTensor)

    def cpu(self):
        return self._type(torch.FloatTensor)

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

        subsample_fourier = self.subsample_fourier
        modulus = self.modulus

        S = input.new(input.size(0),
                      input.size(1),
                      1 + self.L*J + self.L*self.L*J*(J - 1) // 2,
                      self.M_padded//(2**J)-2,
                      self.N_padded//(2**J)-2)
        U_r = pad(input, self.pre_pad)
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
