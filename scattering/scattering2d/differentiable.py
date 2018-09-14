from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from .FFT import fft_c2c, ifft_c2r, ifft_c2c


def prepare_padding_size(M, N, J):
    M_padded = ((M + 2 ** (J))//2**J+1)*2**J
    N_padded = ((N + 2 ** (J))//2**J+1)*2**J

    return M_padded, N_padded


def cast(Psi, Phi, _type):
    for key, item in enumerate(Psi):
        for key2, item2 in Psi[key].items():
            if torch.is_tensor(item2):
                Psi[key][key2] = Variable(item2.type(_type))
    Phi = [Variable(v.type(_type)) for v in Phi]
    return Psi, Phi


def pad(input, J):
    out_ = F.pad(input, (2**J,) * 4, mode='reflect').unsqueeze(input.dim())
    return torch.cat([out_, Variable(input.data.new(out_.size()).zero_())], 4)


def unpad(input):
    return input[..., 1:-1, 1:-1]


def cdgmm(A, B):
    C = Variable(A.data.new(A.size()))

    A_r = A[..., 0].contiguous().view(-1, A.size(-2)*A.size(-3))
    A_i = A[..., 1].contiguous().view(-1, A.size(-2)*A.size(-3))

    B_r = B[...,0].contiguous().view(B.size(-2)*B.size(-3)).unsqueeze(0).expand_as(A_i)
    B_i = B[..., 1].contiguous().view(B.size(-2)*B.size(-3)).unsqueeze(0).expand_as(A_r)

    C[..., 0] = (A_r * B_r - A_i * B_i).view(A.shape[:-1])
    C[..., 1] = (A_r * B_i + A_i * B_r).view(A.shape[:-1])
    return C


def periodize(input, k):
    return input.view(input.size(0), input.size(1),
                      k, input.size(2) // k,
                      k, input.size(3) // k,
                      2).mean(4).squeeze(4).mean(2).squeeze(2)


def modulus(input):
    norm = input.norm(p=2, dim=-1, keepdim=True)
    return torch.cat([norm, Variable(norm.data.new(norm.size()).zero_())], -1)


def scattering(input, psi, phi, J):
    M, N = input.size(-2), input.size(-1)
    M_padded, N_padded = prepare_padding_size(M, N, J)
    S = Variable(input.data.new(input.size(0),
                                input.size(1),
                                1 + 8*J + 8*8*J*(J - 1) // 2,
                                M_padded//(2**J)-2,
                                N_padded//(2**J)-2))
    U_r = pad(input, J)
    U_0_c = fft_c2c(U_r)

    # First low pass filter
    U_1_c = periodize(cdgmm(U_0_c, phi[0]), k=2**J)

    U_J_r = ifft_c2r(U_1_c)

    n = 0
    S[..., n, :, :] = unpad(U_J_r)
    n = n + 1

    for n1 in range(len(psi)):
        j1 = psi[n1]['j']
        U_1_c = cdgmm(U_0_c, psi[n1][0])
        if(j1 > 0):
            U_1_c = periodize(U_1_c, k=2 ** j1)
        U_1_c = fft_c2c(modulus(ifft_c2c(U_1_c)))

        # Second low pass filter
        U_2_c = periodize(cdgmm(U_1_c, phi[j1]), k=2**(J-j1))
        U_J_r = ifft_c2r(U_2_c)
        S[..., n, :, :] = unpad(U_J_r)
        n = n + 1

        for n2 in range(len(psi)):
            j2 = psi[n2]['j']
            if(j1 < j2):
                U_2_c = periodize(cdgmm(U_1_c, psi[n2][j1]), k=2 ** (j2-j1))
                U_2_c = fft_c2c(modulus(ifft_c2c(U_2_c)))

                # Third low pass filter
                U_2_c = periodize(cdgmm(U_2_c, phi[j2]), k=2 ** (J-j2))
                U_J_r = ifft_c2r(U_2_c)

                S[..., n, :, :] = unpad(U_J_r)
                n = n + 1
    return S



