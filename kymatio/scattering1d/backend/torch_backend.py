import torch
import torch.nn.functional as F
from ...backend.torch_backend import TorchBackend

from packaging import version

if version.parse(torch.__version__) >= version.parse('1.8'):
    _fft = lambda x: torch.view_as_real(torch.fft.fft(torch.view_as_complex(x)))
    _ifft = lambda x: torch.view_as_real(torch.fft.ifft(torch.view_as_complex(x)))
    _irfft = lambda x: torch.fft.ifft(torch.view_as_complex(x)).real[..., None]
else:
    _fft = lambda x: torch.fft(x, 1, normalized=False)
    _ifft = lambda x: torch.ifft(x, 1, normalized=False)
    _irfft = lambda x: torch.irfft(x, 1, normalized=False, onesided=False)[..., None]


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

    # we cast to complex here then fft rather than use torch.rfft as torch.rfft is
    # inefficent.
    @classmethod
    def rfft(cls, x):
        cls.contiguous_check(x)
        cls.real_check(x)

        x_r = torch.zeros(x.shape[:-1] + (2,), dtype=x.dtype, layout=x.layout, device=x.device)
        x_r[..., 0] = x[..., 0]

        return _fft(x_r)

    @classmethod
    def irfft(cls, x):
        cls.contiguous_check(x)
        cls.complex_check(x)

        return _irfft(x)

    @classmethod
    def ifft(cls, x):
        cls.contiguous_check(x)
        cls.complex_check(x)

        return _ifft(x)


backend = TorchBackend1D
