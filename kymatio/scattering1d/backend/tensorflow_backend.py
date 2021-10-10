import tensorflow as tf

from ...backend.tensorflow_backend import TensorFlowBackend
from . import agnostic_backend as agnostic


class TensorFlowBackend1D(TensorFlowBackend):
    @classmethod
    def subsample_fourier(cls, x, k, axis=-1):
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
        axis : int
            Axis along which to subsample.

        Returns
        -------
        res : tensor
            The input tensor periodized along the next to last axis to yield a
            tensor of size x.shape[-2] // k along that dimension.
        """
        cls.complex_check(x)

        axis = axis if axis >= 0 else x.ndim + axis  # ensure positive
        s = list(x.shape)
        N = s[axis]
        re = (k, N // k)
        s.pop(axis)
        s.insert(axis, re[1])
        s.insert(axis, re[0])

        y = tf.reshape(x, s)

        return tf.reduce_mean(y, axis=axis)

    @staticmethod
    def pad(x, pad_left, pad_right, pad_mode='reflect', axis=-1):
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
        axis : int
            Axis to pad.
        pad_mode : str
            name of padding to use.

        Returns
        -------
        res : tensor
            The tensor passed along the third dimension.
        """
        return agnostic.pad(x, pad_left, pad_right, pad_mode, axis=axis)

    @staticmethod
    def unpad(x, i0, i1, axis=-1):
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
        axis : int
            Axis to unpad.

        Returns
        -------
        x_unpadded : tensor
            The tensor x[..., i0:i1].
        """
        return x[agnostic.index_axis(i0, i1, axis, x.ndim)]

    @classmethod
    def fft(cls, x, axis=-1):
        x = cls._maybe_transpose_for_fft(x, axis)

        out = tf.signal.fft(x, name='fft1d')
        return cls._maybe_transpose_for_fft(out, axis)

    @classmethod
    def rfft(cls, x, axis=-1):
        cls.real_check(x)
        x = cls._maybe_transpose_for_fft(x, axis)

        out = tf.signal.fft(tf.cast(x, tf.complex64), name='rfft1d')
        return cls._maybe_transpose_for_fft(out, axis)

    @classmethod
    def irfft(cls, x, axis=-1):
        cls.complex_check(x)
        x = cls._maybe_transpose_for_fft(x, axis)

        out = tf.math.real(tf.signal.ifft(x, name='irfft1d'))
        return cls._maybe_transpose_for_fft(out, axis)

    @classmethod
    def ifft(cls, x, axis=-1):
        cls.complex_check(x)
        x = cls._maybe_transpose_for_fft(x, axis)

        out = tf.signal.ifft(x, name='ifft1d')
        return cls._maybe_transpose_for_fft(out, axis)

    @classmethod
    def conj_reflections(cls, x, ind_start, ind_end, k, N, pad_left, pad_right,
                         trim_tm):
        return agnostic.conj_reflections(cls, x, ind_start, ind_end, k, N,
                                         pad_left, pad_right, trim_tm)

    @classmethod
    def _maybe_transpose_for_fft(cls, x, axis):
        if axis in (-2, x.ndim - 2) and x.ndim > 2:
            D = x.ndim
            x = tf.transpose(x, (*list(range(D - 2)), D - 1, D - 2))
        elif axis not in (-1, x.ndim - 1):
            # -1 means no need to transpose
            raise NotImplementedError("`axis` must be -1 or -2")
        return x

backend = TensorFlowBackend1D
