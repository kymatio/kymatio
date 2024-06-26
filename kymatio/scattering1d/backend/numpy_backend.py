from ...backend.numpy_backend import NumpyBackend


class NumpyBackend1D(NumpyBackend):
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

        y = x.reshape(x.shape[:-1] + (k, x.shape[-1] // k))

        res = y.mean(axis=-2)

        return res

    @classmethod
    def pad(cls, x, pad_left, pad_right, mode='reflect'):
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
        output : tensor
            The tensor passed along the third dimension.
        """
        if mode != 'constant':
            if (pad_left >= x.shape[-1]) or (pad_right >= x.shape[-1]):
                raise ValueError('Indefinite padding size (larger than tensor).')

        paddings = ((0, 0),) * len(x.shape[:-1])
        paddings += (pad_left, pad_right),

        output = cls._np.pad(x, paddings, mode=mode)

        return output

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
        return x[..., i0:i1]

    @classmethod
    def cfft(cls, x):
        cls.complex_check(x)

        return cls._np.fft.fft(x)

    @classmethod
    def rfft(cls, x):
        cls.real_check(x)

        return cls._np.fft.fft(x)

    @classmethod
    def irfft(cls, x):
        cls.complex_check(x)

        return cls._fft.ifft(x).real

    @classmethod
    def ifft(cls, x):
        cls.complex_check(x)

        return cls._fft.ifft(x)

    @classmethod
    def average_global(cls, x):
        cls.real_check(x)

        return cls._np.sum(x, axis=-1, keepdims=True)

    @classmethod
    def pad_frequency(cls, x, padding):
        return cls.pad(x, pad_left=0, pad_right=padding, mode='constant')

    @classmethod
    def swap_time_frequency(cls, x):
        return cls._np.moveaxis(x, source=(-1, -2), destination=(-2, -1))

    @classmethod
    def unpad_frequency(cls, x, n1_max, n1_stride):
        n1_unpadded = 1 + (n1_max // n1_stride)
        return x[..., :n1_unpadded, :]

    @classmethod
    def split_frequency_axis(cls, x):
        return cls._np.split(x, indices_or_sections=x.shape[-2], axis=-2)

backend = NumpyBackend1D
