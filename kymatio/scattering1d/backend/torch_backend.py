import torch
import torch.fft
from ...backend.torch_backend import TorchBackend
from .agnostic_backend import pad as agnostic_pad, index_axis


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

        N = x.shape[-1]

        x = torch.view_as_real(x)
        res = x.view(x.shape[:-2] + (k, N // k, 2)).mean(dim=-3)
        res = torch.view_as_complex(res)

        return res

    @staticmethod
    def pad(x, pad_left, pad_right, axis=-1, pad_mode='reflect'):
        """Pad N-dim tensor.

        N-dim implementation of the padding function for real PyTorch tensors.

        Parameters
        ----------
        x : tensor
            Input with at least one axis.
        pad_left : int
            Amount to add on the left of the tensor (at the beginning of the
            temporal axis).
        pad_right : int
            amount to add on the right of the tensor (at the end of the temporal
            axis).
        axis : int
            Axis to pad.
        pad_mode : str
            name of padding to use.

        Returns
        -------
        res : tensor
            The tensor passed along the third dimension.
        """
        return agnostic_pad(x, pad_left, pad_right, pad_mode, axis, 'torch')

    @staticmethod
    def unpad(x, i0, i1, axis=-1):
        """Unpad N-dim tensor.

        Slices the input tensor at indices between i0 and i1 along any axis.

        Parameters
        ----------
        x : tensor
            Input with at least one axis.
        i0 : int
            Start of original signal before padding.
        i1 : int
            End of original signal before padding.
        axis : int
            Axis to unpad.

        Returns
        -------
        x_unpadded : tensor
            The tensor x[..., i0:i1].
        """
        return x[index_axis(i0, i1, axis, x.ndim)]

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
    def rfft(cls, x, axis=-1):
        cls.contiguous_check(x)
        cls.real_check(x)

        return torch.fft.fft(x, dim=axis)

    @classmethod
    def irfft(cls, x, axis=-1):
        cls.contiguous_check(x)
        cls.complex_check(x)

        return torch.fft.ifft(x, dim=axis).real

    @classmethod
    def ifft(cls, x, axis=-1):
        cls.contiguous_check(x)
        cls.complex_check(x)

        return torch.fft.ifft(x, dim=axis)

    @classmethod
    def mean(cls, x, axis=-1):
        """Take mean along specified axis, without collapsing the axis."""
        return x.mean(axis, keepdim=True)


backend = TorchBackend1D
