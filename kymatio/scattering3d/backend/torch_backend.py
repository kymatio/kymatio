import torch
import warnings
from torch.nn.functional import pad
BACKEND_NAME = 'torch'
from collections import namedtuple

from ...backend.torch_backend import cdgmm, contiguous_check, Modulus, concatenate, complex_check, real_check

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


class Pad(object):
    def __init__(self, pad_size, input_size, pre_pad=False):
        """Padding which allows to simultaneously pad in a reflection fashion
        and map to complex. 

        Parameters
        ----------
        pad_size : list of 4 integers
            Size of padding to apply [top, bottom, left, right].
        input_size : list of 2 integers
            Size of the original signal [height, width].
        pre_pad : boolean, optional
            If set to true, then there is no padding one simply adds the
            imaginary part. 

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
        if pad_size_tmp[4] == self.input_size[2]:
            pad_size_tmp[4] -= 1
            pad_size_tmp[5] -= 1

        self.pad_size_tmp = pad_size_tmp

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
        if not self.pre_pad:
            # Pytorch expects its padding as [left, right, top, bottom]
            pad_size_tmp = self.pad_size_tmp
            x = pad(x, (pad_size_tmp[0], pad_size_tmp[1], 
                pad_size_tmp[2], pad_size_tmp[3], pad_size_tmp[4],
                pad_size_tmp[5]), mode='constant')

            # Note: PyTorch is not effective to pad signals of size N-1 with N
            # elements, thus we had to add this fix.
            if self.pad_size[0] == self.input_size[0]:
                x = torch.cat([x[:, 1, :, :, :, :].unsqueeze(2), x,
                    x[:, x.shape[1] - 2, :, :, :, :].unsqueeze(2)], 2)
            if self.pad_size[2] == self.input_size[1]:
                x = torch.cat([x[:, :, 1, :, :, :].unsqueeze(3), x, 
                    x[:, :, :, x.shape[3] - 2, :, :].unsqueeze(3)], 3)
            if self.pad_size[4] == self.input_size[2]:
                x = torch.cat([x[:, :, :, :, 1, :].unsqueeze(4), x, 
                    x[:, :, :, :, x.shape[4] - 2, :].unsqueeze(4)], 4)

            return x.unsqueeze(-1)

def unpad(in_):
    """Unpads input.

        Slices the input tensor at indices between 1:-1.

        Parameters
        ----------
        in_ : tensor
            Input tensor.

        Returns
        -------
        in_[..., 1:-1, 1:-1, 1:-1, :] : tensor
            Output tensor. Unpadded input.

    """
    return in_[..., 1:-1, 1:-1, 1:-1, 0]

class SubsampleFourier(object):
    def __call__(self, x, k):
        complex_check(x)
        contiguous_check(x)

        y = x.view(-1,
                    k, x.shape[1] // k,
                    k, x.shape[2] // k,
                    k, x.shape[3] // k,
                    2)
        out = y.mean((5, 3, 1), keepdim=False)
        return out



        

def concatenate(arrays, L):
    S = torch.stack(arrays, dim=1)
    S = S.reshape((S.shape[0], S.shape[1] // (L + 1), (L + 1)) + S.shape[2:])
    return S

def concatenate_3d(x):
    return torch.stack(x, 1)

# we cast to complex here then fft rather than use torch.rfft as torch.rfft is
# inefficent.
def rfft(x):
    contiguous_check(x)
    real_check(x)

    x_r = torch.zeros((x.shape[:-1] + (2,)), dtype=x.dtype, layout=x.layout, device=x.device)
    x_r[..., 0] = x[..., 0]

    return torch.fft(x_r, 3, normalized=False)


def irfft(x):
    contiguous_check(x)
    complex_check(x)

    return torch.ifft(x, 3, normalized=False)[..., :1]


def ifft(x):
    contiguous_check(x)
    complex_check(x)

    return torch.ifft(x, 3, normalized=False)



backend = namedtuple('backend',
                     ['name',
                      'cdgmm3d',
                      'fft',
                      'modulus',
                      'modulus_rotation',
                      'compute_integrals',
                      'concatenate'])

backend.name = 'torch'
backend.cdgmm3d = cdgmm
backend.rfft = rfft
backend.irfft = irfft
backend.ifft = ifft
backend.concatenate = concatenate
backend.concatenate_3d = concatenate_3d
backend.modulus = Modulus()
backend.modulus_rotation = modulus_rotation
backend.compute_integrals = compute_integrals
backend.Pad = Pad
backend.unpad = unpad
backend.subsample_fourier = SubsampleFourier()
