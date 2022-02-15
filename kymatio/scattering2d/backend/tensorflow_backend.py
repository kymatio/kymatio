import tensorflow as tf

from ...backend.tensorflow_backend import TensorFlowBackend


class Pad(object):
    def __init__(self, pad_size, input_size):
        """
            Padding which allows to simultaneously pad in a reflection fashion
            and map to complex.
            Parameters
            ----------
            pad_size : list of 4 integers
                size of padding to apply.
            input_size : list of 2 integers
                size of the original signal
        """
        self.pad_size = pad_size

    def __call__(self, x):
        paddings = [[0, 0]] * len(x.shape[:-2])
        paddings += [[self.pad_size[0], self.pad_size[1]], [self.pad_size[2], self.pad_size[3]]]
        return tf.pad(x, paddings, mode="REFLECT")


class TensorFlowBackend2D(TensorFlowBackend):
    Pad = Pad

    @staticmethod
    def unpad(in_):
        """
            Slices the input tensor at indices between 1::-1
            Parameters
            ----------
            in_ : tensor_like
                input tensor
            Returns
            -------
            in_[..., 1:-1, 1:-1]
        """
        return in_[..., 1:-1, 1:-1]


    @classmethod
    def rfft(cls, x):
        cls.real_check(x)
        return tf.signal.fft2d(tf.cast(x, tf.complex64), name='rfft2d')

    @classmethod
    def irfft(cls, x):
        cls.complex_check(x)
        return tf.math.real(tf.signal.ifft2d(x, name='irfft2d'))


    @classmethod
    def ifft(cls, x):
        cls.complex_check(x)
        return tf.signal.ifft2d(x, name='ifft2d')

    @classmethod
    def subsample_fourier(cls, x, k):
        """ Subsampling of a 2D image performed in the Fourier domain.

        Subsampling in the spatial domain amounts to periodization
        in the Fourier domain, hence the formula.

        Parameters
        ----------
        x : tensor_like
            input tensor with at least three dimensions.
        k : int
            integer such that x is subsampled by k along the spatial variables.

        Returns
        -------
        out : tensor_like
            Tensor such that its Fourier transform is the Fourier
            transform of a subsampled version of x, i.e. in
            F^{-1}(out)[u1, u2] = F^{-1}(x)[u1 * k, u2 * k]

        """
        cls.complex_check(x)

        y = tf.reshape(x, (-1, k, x.shape[1] // k, k, x.shape[2] // k))

        out = tf.reduce_mean(y, axis=(1, 3))
        return out


backend = TensorFlowBackend2D
