import torch
from torch.nn import ReflectionPad2d
from collections import namedtuple

BACKEND_NAME = 'torch'

#from ...backend.torch_backend import cdgmm, contiguous_check, Modulus, concatenate, complex_check, real_check

from ...backend.torch_backend import TorchBackend


class Pad(object):
    def __init__(self, pad_size, input_size, pre_pad=False):
        """Padding which allows to simultaneously pad in a reflection fashion
            and map to complex.

            Parameters
            ----------
            pad_size : list of 4 integers
                Size of padding to apply [top, bottom, left, right].
            input_size : list of 2 integers
                size of the original signal [height, width].
            pre_pad : boolean, optional
                If set to true, then there is no padding, one simply adds the imaginary part.

        """
        self.pre_pad = pre_pad
        self.pad_size = pad_size
        self.input_size = input_size

        self.build()

    def build(self):
        """Builds the padding module.

            Attributes
            ----------
            padding_module : ReflectionPad2d
                Pads the input tensor using the reflection of the input
                boundary.

        """
        pad_size_tmp = list(self.pad_size)

        # This handles the case where the padding is equal to the image size
        if pad_size_tmp[0] == self.input_size[0]:
            pad_size_tmp[0] -= 1
            pad_size_tmp[1] -= 1
        if pad_size_tmp[2] == self.input_size[1]:
            pad_size_tmp[2] -= 1
            pad_size_tmp[3] -= 1
        # Pytorch expects its padding as [left, right, top, bottom]
        self.padding_module = ReflectionPad2d([pad_size_tmp[2], pad_size_tmp[3],
                                               pad_size_tmp[0], pad_size_tmp[1]])

    def __call__(self, x):
        """Applies padding and maps to complex.

            Parameters
            ----------
            x : tensor
                Real tensor input to be padded and sent to complex domain.

            Returns
            -------
            output : tensor
                Complex torch tensor that has been padded.

        """
        batch_shape = x.shape[:-2]
        signal_shape = x.shape[-2:]
        x = x.reshape((-1, 1) + signal_shape)
        if not self.pre_pad:
            x = self.padding_module(x)

            # Note: PyTorch is not effective to pad signals of size N-1 with N
            # elements, thus we had to add this fix.
            if self.pad_size[0] == self.input_size[0]:
                x = torch.cat([x[:, :, 1, :].unsqueeze(2), x, x[:, :, x.shape[2] - 2, :].unsqueeze(2)], 2)
            if self.pad_size[2] == self.input_size[1]:
                x = torch.cat([x[:, :, :, 1].unsqueeze(3), x, x[:, :, :, x.shape[3] - 2].unsqueeze(3)], 3)

        output = x.reshape(batch_shape + x.shape[-2:] + (1,))
        return output

def unpad(in_):
    """Unpads input.

        Slices the input tensor at indices between 1:-1.

        Parameters
        ----------
        in_ : tensor
            Input tensor.

        Returns
        -------
        in_[..., 1:-1, 1:-1] : tensor
            Output tensor.  Unpadded input.

    """
    in_ = in_[..., 1:-1, 1:-1, :]
    in_ = in_.reshape(in_.shape[:-1])
    return in_

class SubsampleFourier(object):
    """Subsampling of a 2D image performed in the Fourier domain

        Subsampling in the spatial domain amounts to periodization
        in the Fourier domain, hence the formula.

        Parameters
        ----------
        x : tensor
            Input tensor with at least 5 dimensions, the last being the real
            and imaginary parts.
        k : int
            Integer such that x is subsampled by k along the spatial variables.

        Returns
        -------
        out : tensor
            Tensor such that its Fourier transform is the Fourier
            transform of a subsampled version of x, i.e. in
            F^{-1}(out)[u1, u2] = F^{-1}(x)[u1 * k, u2 * k].

    """
    def __call__(self, x, k):
        complex_check(x)
        contiguous_check(x)

        batch_shape = x.shape[:-3]
        signal_shape = x.shape[-3:]
        x = x.view((-1,) + signal_shape)
        y = x.view(-1,
                       k, x.shape[1] // k,
                       k, x.shape[2] // k,
                       2)

        out = y.mean(3, keepdim=False).mean(1, keepdim=False)
        out = out.reshape(batch_shape + out.shape[-3:])
        return out


# we cast to complex here then fft rather than use torch.rfft as torch.rfft is
# inefficent.
def rfft(x):
    contiguous_check(x)
    real_check(x)

    x_r = torch.zeros((x.shape[:-1] + (2,)), dtype=x.dtype, layout=x.layout, device=x.device)
    x_r[..., 0] = x[..., 0]

    return torch.fft(x_r, 2, normalized=False)


def irfft(x):
    contiguous_check(x)
    complex_check(x)

    return torch.ifft(x, 2, normalized=False)[..., :1]


def ifft(x):
    contiguous_check(x)
    complex_check(x)

    return torch.ifft(x, 2, normalized=False)


def concatenate_2d(x):
    return concatenate(x, -3)


#backend = namedtuple('backend', ['name', 'cdgmm', 'modulus', 'subsample_fourier', 'fft', 'Pad', 'unpad', 'concatenate'])

backend = TorchBackend()
#backend.name = 'torch'
#backend.cdgmm = cdgmm
#backend.modulus = Modulus()
backend.subsample_fourier = SubsampleFourier()
backend.rfft = rfft
backend.irfft = irfft
backend.ifft = ifft
backend.Pad = Pad
backend.unpad = unpad

contiguous_check = backend.contiguous_check
complex_check = backend.complex_check
real_check = backend.real_check
concatenate = backend.concatenate
backend.concatenate = concatenate_2d
