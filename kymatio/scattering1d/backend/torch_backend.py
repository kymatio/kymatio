import torch
import torch.nn.functional as F
from ...backend.torch_backend import TorchBackend

from packaging import version

if version.parse(torch.__version__) >= version.parse("1.8"):
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
    def pad(x, pad_left, pad_right, mode="reflect"):
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
        mode : string (optional)
            padding mode: "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive)

        Returns
        -------
        res : tensor
            The tensor passed along the third dimension.
        """
        if mode != "constant":
            if (pad_left >= x.shape[-1]) or (pad_right >= x.shape[-1]):
                raise ValueError("Indefinite padding size (larger than tensor).")

        res = F.pad(x, (pad_left, pad_right), mode=mode)
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
    def cfft(cls, x):
        cls.contiguous_check(x)
        cls.complex_check(x)

        return _fft(x)

    # we cast to complex here then fft rather than use torch.rfft as torch.rfft is
    # inefficent.
    @classmethod
    def rfft(cls, x):
        cls.contiguous_check(x)
        cls.real_check(x)

        x_r = torch.zeros(
            x.shape[:-1] + (2,), dtype=x.dtype, layout=x.layout, device=x.device
        )
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

    @classmethod
    def average_global(cls, x):
        cls.contiguous_check(x)
        cls.real_check(x)

        return torch.sum(x, axis=-2, keepdims=True)

    @classmethod
    def pad_frequency(cls, x, padding):
        """Pad the frequency axis in preparation for frequency scattering.
        pad_frequency has some important differences with pad:
        1. in pad_frequency, the input Tensor has a trailing singleton
        dimension to represent real vs. imaginary dimensions. Hence,
        pad_frequency operates over the penultimate dimension whereas pad
        operates over the last dimension.
        2. pad_frequency does not add a trailing singleton dimmension to its
        output. This is unlike pad and for the same reasons as (1).
        3. pad_frequency is one-sided. It extends the frequency axis to the
        "right", i.e., to lower frequencies. This is because "right" translates
        to higher values of the psi1 wavelet index n1 and thus to lower values
        of the center frequency xi1.
        4. pad_frequency is 'constant' whereas 'pad' is 'reflect' by default.
        This is for reasons of energy preservation, and also because there is
        no reason why the reflect power spectral density near Nyquist to be a
        continuation of the power spectral density near bin n1=_N_padded_fr.
        """
        return F.pad(x, (0, 0, 0, padding), mode="constant", value=0)

    @classmethod
    def swap_time_frequency(cls, x):
        """Swap time and frequency dimensions of a tensor
        Parameters
        ----------
        x : tensor
            if complex: (batch, frequency, time, real/imag)
            else: (batch, frequency, time, 1)

        Returns
        -------
        output : tensor
            if complex: (batch, time, frequency, real/imag)
            else: (batch, time, frequency, 1)
        """
        return torch.transpose(x, dim0=-2, dim1=-3).contiguous()

    @staticmethod
    def unpad_frequency(x, n1_max, n1_stride):
        """Unpad the frequency axis after frequency scattering and/or frequency
        averaging. This is called at the end of `jtfs_average_and_format`
        unless `out_type='array'` and `format='joint'`.
        NB. Unpadding is one-sided. See point 3 of pad_frequency docstring.

        Parameters
        ----------
        x : tensor (batch, frequency, time, 1), corresponds to path['coef']
        n1_max: integer, the last first-order wavelet index (n1) such that j1 < j2
            for the scattering path of x. By definition, lower than len(psi1_f).
        n1_stride: integer frequential subsampling factor associated to the
            scattering path of x. Equal to max(1, 2**(j_fr - oversampling_fr)).

        Returns
        -------
        output : tensor (batch, time, unpadded frequency, 1)
        """
        n1_unpadded = 1 + (n1_max // n1_stride)
        return x[:, :, :n1_unpadded, :]

    @staticmethod
    def split_frequency_axis(x):
        """Split tensor along its frequency axis.

        Parameters
        ----------
        x : tensor (batch, frequency, time, 1), corresponds to path['coef']

        Returns
        -------
        output : list of tensors, each of shape (batch, 1, time, 1). The number
        of elements in the list is equal to that of the frequency axis.
        """
        return torch.split(x, 1, dim=-3)


backend = TorchBackend1D
