import torch
import warnings

from collections import namedtuple
from packaging import version

from ...backend.torch_backend import TorchBackend


if version.parse(torch.__version__) >= version.parse('1.8'):
    def _fft(input, inverse=False):
        """Interface with torch FFT routines for 3D signals.
            fft of a 3d signal
            Example
            -------
            x = torch.randn(128, 32, 32, 32, 2)

            x_fft = fft(x)
            x_ifft = fft(x, inverse=True)
            Parameters
            ----------
            x : tensor
                Complex input for the FFT.
            inverse : bool
                True for computing the inverse FFT.

            Raises
            ------
            TypeError
                In the event that x does not have a final dimension 2 i.e. not
                complex.

            Returns
            -------
            output : tensor
                Result of FFT or IFFT.
        """
        if inverse:
            return torch.view_as_real(torch.fft.ifftn(torch.view_as_complex(input), dim=[-1, -2, -3]))
        return torch.view_as_real(torch.fft.fftn(torch.view_as_complex(input), dim=[-1, -2, -3]))
else:
    def _fft(input, inverse=False):
        """Interface with torch FFT routines for 3D signals.
            fft of a 3d signal
            Example
            -------
            x = torch.randn(128, 32, 32, 32, 2)
            x_fft = fft(x)
            x_ifft = fft(x, inverse=True)
            Parameters
            ----------
            x : tensor
                Complex input for the FFT.
            inverse : bool
                True for computing the inverse FFT.
            Raises
            ------
            TypeError
                In the event that x does not have a final dimension 2 i.e. not
                complex.
            Returns
            -------
            output : tensor
                Result of FFT or IFFT.
        """
        if inverse:
            return torch.ifft(input, 3)
        return torch.fft(input, 3)


class TorchBackend3D(TorchBackend):
    @staticmethod
    def stack(arrays, L):
        S = torch.stack(arrays, dim=1)
        S = S.reshape((S.shape[0], S.shape[1] // (L + 1), (L + 1)) + S.shape[2:])
        return S

    # we cast to complex here then fft rather than use torch.rfft as torch.rfft is
    # inefficent.
    @classmethod
    def rfft(cls, x):
        cls.contiguous_check(x)
        cls.real_check(x)

        x_r = torch.zeros((x.shape[:-1] + (2,)), dtype=x.dtype, layout=x.layout, device=x.device)
        x_r[..., 0] = x[..., 0]

        return _fft(x_r)

    @classmethod
    def ifft(cls, x):
        cls.contiguous_check(x)
        cls.complex_check(x)

        return _fft(x, inverse=True)

    @classmethod
    def cdgmm3d(cls, A, B):
        return cls.cdgmm(A, B)

    @staticmethod
    def modulus_rotation(x, module=None):
        """Used for computing rotation invariant scattering transform coefficents.

            Parameters
            ----------
            x : tensor
                Size (batchsize, M, N, O, 2).
            module : tensor
                Tensor that holds the overall sum. If none, initializes the tensor
                to zero (default).
            Returns
            -------
            output : torch tensor
                Tensor of the same size as input_array. It holds the output of
                the operation::
                $\\sqrt{\\sum_m (\\text{input}_\\text{array} \\star \\psi_{j,l,m})^2)}$
                which is covariant to 3D translations and rotations.
        """
        if module is None:
            module = (x ** 2).sum(-1, keepdim=True)
        else:
            module = module ** 2 + (x ** 2).sum(-1, keepdim=True)
        return torch.sqrt(module)

    @staticmethod
    def compute_integrals(input_array, integral_powers):
        """Computes integrals.

            Computes integrals of the input_array to the given powers.
            Parameters
            ----------
            input_array : torch tensor
                Size (B, M, N, O), where B is batch_size, and M, N, O are spatial
                dims.
            integral_powers : list
                List of P positive floats containing the p values used to
                compute the integrals of the input_array to the power p (l_p
                norms).
            Returns
            -------
            integrals : torch tensor
                Tensor of size (B, P) containing the integrals of the input_array
                to the powers p (l_p norms).
        """
        integrals = torch.zeros((input_array.shape[0], len(integral_powers)),
                device=input_array.device)
        for i_q, q in enumerate(integral_powers):
            integrals[:, i_q] = (input_array ** q).view(
                input_array.shape[0], -1).sum(1)
        return integrals


backend = TorchBackend3D
