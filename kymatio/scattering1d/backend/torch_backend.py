import torch
import torch.fft
import torch.nn.functional as F
from ...backend.torch_backend import TorchBackend


class TorchBackend1D(TorchBackend):
    @classmethod
    def subsample_fourier(cls, x, k):
        """Subsampling in the Fourier domain

        Subsampling in the temporal domain amounts to periodization in the Fourier
        domain, so the input is periodized according to the subsampling factor.

        Parameters
        ----------
        x : tensor
            Input tensor with at least 3 dimensions, where the next to last
            corresponds to the frequency index in the standard PyTorch FFT
            ordering. The length of this dimension should be a power of 2 to
            avoid errors. The last dimension should represent the real and
            imaginary parts of the Fourier transform.
        k : int
            The subsampling factor.

        Returns
        -------
        res : tensor
            The input tensor periodized along the next to last axis to yield a
            tensor of size x.shape[-2] // k along that dimension.
        """
        cls.complex_check(x)

        N = x.shape[-2]

        res = x.view(x.shape[:-2] + (k, N // k, 2)).mean(dim=-3)

        return res

    @staticmethod
    def pad(x, pad_left, pad_right):
        """Pad real 1D tensors

        1D implementation of the padding function for real PyTorch tensors.

        Parameters
        ----------
        x : tensor
            Three-dimensional input tensor with the third axis being the one to
            be padded.
        pad_left : int
            Amount to add on the left of the tensor (at the beginning of the
            temporal axis).
        pad_right : int
            amount to add on the right of the tensor (at the end of the temporal
            axis).
        Returns
        -------
        res : tensor
            The tensor passed along the third dimension.
        """
        if (pad_left >= x.shape[-1]) or (pad_right >= x.shape[-1]):
            raise ValueError('Indefinite padding size (larger than tensor).')

        res = F.pad(x, (pad_left, pad_right), mode='reflect')
        res = res[..., None]

        return res

    @staticmethod
    def unpad(x, i0, i1):
        """Unpad real 1D tensor

        Slices the input tensor at indices between i0 and i1 along the last axis.

        Parameters
        ----------
        x : tensor
            Input tensor with least one axis.
        i0 : int
            Start of original signal before padding.
        i1 : int
            End of original signal before padding.

        Returns
        -------
        x_unpadded : tensor
            The tensor x[..., i0:i1].
        """
        x = x.reshape(x.shape[:-1])

        return x[..., i0:i1]

    @classmethod
    def zeros_like(cls, ref, shape=None):
        shape = shape if shape is not None else ref.shape
        return torch.zeros(shape, dtype=ref.dtype, layout=ref.layout,
                           device=ref.device)

    @classmethod
    def fft(cls, x, axis=-1):
        cls.contiguous_check(x)
        return torch.fft.fft(x, dim=axis)

    # we cast to complex here then fft rather than use torch.rfft as torch.rfft is
    # inefficent.
    @classmethod
    def rfft(cls, x):
        cls.contiguous_check(x)
        cls.real_check(x)

        x_r = torch.zeros(x.shape[:-1] + (2,), dtype=x.dtype, layout=x.layout, device=x.device)
        x_r[..., 0] = x[..., 0]

        return torch.fft.fft(x_r, 1)

    @classmethod
    def irfft(cls, x):
        cls.contiguous_check(x)
        cls.complex_check(x)

        return torch.fft.ifft(x, 1, norm='forward')[..., :1]

    @classmethod
    def ifft(cls, x):
        cls.contiguous_check(x)
        cls.complex_check(x)

        return torch.fft.ifft(x, 1, norm='forward')

    @classmethod
    def transpose(cls, x):
        """Permute time and frequency dimension for time-frequency scattering"""
        return x.transpose(-2, -3).contiguous()

    @classmethod
    def conj_fr(cls, x):
        """Conjugate in frequency domain by swapping all bins (except dc);
        assumes frequency along last axis.
        """
        out = cls.zeros_like(x)
        out[..., 0] = x[..., 0]
        out[..., 1:] = x[..., :0:-1]
        return out

    @classmethod
    def mean(cls, x, axis=-1):
        """Take mean along specified axis, without collapsing the axis."""
        return x.mean(axis, keepdim=True)


backend = TorchBackend1D
