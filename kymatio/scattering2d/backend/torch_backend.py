# Authors: Edouard Oyallon, Sergey Zagoruyko

import torch
from torch.nn import ReflectionPad2d
from collections import namedtuple
from packaging import version


BACKEND_NAME = 'torch'

from ...backend.torch_backend import _is_complex, cdgmm, type_checks, Modulus, concatenate
from ...backend.base_backend import FFT


class Pad(object):
    def __init__(self, pad_size, input_size):
        """Padding which allows to simultaneously pad in a reflection fashion
            and map to complex.

            Parameters
            ----------
            pad_size : list of 4 integers
                Size of padding to apply [top, bottom, left, right].
            input_size : list of 2 integers
                size of the original signal [height, width].
        """
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
        x = self.padding_module(x)

        # Note: PyTorch is not effective to pad signals of size N-1 with N
        # elements, thus we had to add this fix.
        if self.pad_size[0] == self.input_size[0]:
            x = torch.cat([x[:, :, 1, :].unsqueeze(2), x, x[:, :, x.shape[2] - 2, :].unsqueeze(2)], 2)
        if self.pad_size[2] == self.input_size[1]:
            x = torch.cat([x[:, :, :, 1].unsqueeze(3), x, x[:, :, :, x.shape[3] - 2].unsqueeze(3)], 3)

        output = x.new_zeros(x.shape + (2,))
        output[..., 0] = x
        output = output.reshape(batch_shape + output.shape[-3:])
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
    return in_[..., 1:-1, 1:-1]

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
        if not _is_complex(x):
            raise TypeError('The x should be complex.')

        if not x.is_contiguous():
            raise RuntimeError('Input should be contiguous.')
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

if version.parse(torch.__version__) >= version.parse('1.8'):
    fft = FFT(lambda x: torch.view_as_real(torch.fft.fft2(torch.view_as_complex(x))),
          lambda x: torch.view_as_real(torch.fft.ifft2(torch.view_as_complex(x))),
          lambda x: torch.fft.ifft2(torch.view_as_complex(x)).real,
          type_checks)
else:
    fft = FFT(lambda x: torch.fft(x, 2, normalized=False),
              lambda x: torch.ifft(x, 2, normalized=False),
              lambda x: torch.irfft(x, 2, normalized=False, onesided=False),
              type_checks)

backend = namedtuple('backend', ['name', 'cdgmm', 'modulus', 'subsample_fourier', 'fft', 'Pad', 'unpad', 'concatenate'])
backend.name = 'torch'
backend.version = torch.__version__
backend.cdgmm = cdgmm
backend.modulus = Modulus()
backend.subsample_fourier = SubsampleFourier()
backend.fft = fft
backend.Pad = Pad
backend.unpad = unpad
backend.concatenate = lambda x: concatenate(x, -3)
