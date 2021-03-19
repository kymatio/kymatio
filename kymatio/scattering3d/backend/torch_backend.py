import torch
import warnings

BACKEND_NAME = 'torch'
from collections import namedtuple
from packaging import version


def _is_complex(input):
    """Checks if input is complex.

        Parameters
        ----------
        input : tensor
            Input to be checked if complex.
        Returns
        -------
        output : boolean
            Returns True if complex (i.e. final dimension is 2), False
            otherwise.
    """
    return input.shape[-1] == 2


def complex_modulus(input_array):
    """Computes complex modulus.

        Parameters
        ----------
        input_array : tensor
            Input tensor whose complex modulus is to be calculated.
        Returns
        -------
        modulus : tensor
            Tensor the same size as input_array. modulus[..., 0] holds the
            result of the complex modulus, modulus[..., 1] = 0.

    """
    modulus = torch.zeros_like(input_array)
    modulus[..., 0] = torch.sqrt((input_array ** 2).sum(-1))
    return modulus


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
        module = torch.zeros_like(x)
    else:
        module = module ** 2
    module[..., 0] += (x ** 2).sum(-1)
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

if version.parse(torch.__version__) >= version.parse('1.8'):
    def fft(input, inverse=False):
        """Interface with torch FFT routines for 3D signals.
            fft of a 3d signal
            Example
            -------
            x = torch.randn(128, 32, 32, 32, 2)

            x_fft = fft(x)
            x_ifft = fft(x, inverse=True)
            Parameters
            ----------
            x : tensor
                Complex input for the FFT.
            inverse : bool
                True for computing the inverse FFT.

            Raises
            ------
            TypeError
                In the event that x does not have a final dimension 2 i.e. not
                complex.

            Returns
            -------
            output : tensor
                Result of FFT or IFFT.
        """
        if not _is_complex(input):
            raise TypeError('The input should be complex (e.g. last dimension is 2)')
        if inverse:
            return torch.view_as_real(torch.fft.ifftn(torch.view_as_complex(input), dim=[-1, -2, -3]))
        return torch.view_as_real(torch.fft.fftn(torch.view_as_complex(input), dim=[-1, -2, -3]))
else:
    def fft(input, inverse=False):
        """Interface with torch FFT routines for 3D signals.
            fft of a 3d signal
            Example
            -------
            x = torch.randn(128, 32, 32, 32, 2)
            x_fft = fft(x)
            x_ifft = fft(x, inverse=True)
            Parameters
            ----------
            x : tensor
                Complex input for the FFT.
            inverse : bool
                True for computing the inverse FFT.
            Raises
            ------
            TypeError
                In the event that x does not have a final dimension 2 i.e. not
                complex.
            Returns
            -------
            output : tensor
                Result of FFT or IFFT.
        """
        if not _is_complex(input):
            raise TypeError('The input should be complex (e.g. last dimension is 2)')
        if inverse:
            return torch.ifft(input, 3)
        return torch.fft(input, 3)

def cdgmm3d(A, B, inplace=False):
    """Complex pointwise multiplication.

        Complex pointwise multiplication between (batched) tensor A and tensor B.

        Parameters
        ----------
        A : torch tensor
            Complex torch tensor.
        B : torch tensor
            Complex of the same size as A.
        inplace : boolean, optional
            If set True, all the operations are performed inplace.

        Raises
        ------
        RuntimeError
            In the event that the tensors are not compatibile for multiplication
            (i.e. the final four dimensions of A do not match with the dimensions
            of B), or in the event that B is not complex, or in the event that the
            type of A and B are not the same.
        TypeError
            In the event that x is not complex i.e. does not have a final dimension
            of 2, or in the event that both tensors are not on the same device.

        Returns
        -------
        output : torch tensor
            Torch tensor of the same size as A containing the result of the
            elementwise complex multiplication of A with B.
    """
    if not A.is_contiguous():
        warnings.warn("cdgmm3d: tensor A is converted to a contiguous array.")
        A = A.contiguous()
    if not B.is_contiguous():
        warnings.warn("cdgmm3d: tensor B is converted to a contiguous array.")
        B = B.contiguous()

    if A.shape[-4:] != B.shape:
        raise RuntimeError('The tensors are not compatible for multiplication.')

    if not _is_complex(A) or not _is_complex(B):
        raise TypeError('The input, filter and output should be complex.')

    if B.ndimension() != 4:
        raise RuntimeError('The second tensor must be simply a complex array.')

    if type(A) is not type(B):
        raise RuntimeError('A and B should be same type.')

    if A.device.type != B.device.type:
        raise TypeError('A and B must be both on GPU or both on CPU.')

    if A.device.type == 'cuda':
        if A.device.index != B.device.index:
            raise TypeError('A and B must be on the same GPU.')

    C = A.new(A.shape)

    C[..., 0] = A[..., 0] * B[..., 0] - A[..., 1] * B[..., 1]
    C[..., 1] = A[..., 0] * B[..., 1] + A[..., 1] * B[..., 0]

    return C if not inplace else A.copy_(C)


def concatenate(arrays, L):
    S = torch.stack(arrays, dim=1)
    S = S.reshape((S.shape[0], S.shape[1] // (L + 1), (L + 1)) + S.shape[2:])
    return S


backend = namedtuple('backend',
                     ['name',
                      'cdgmm3d',
                      'fft',
                      'modulus',
                      'modulus_rotation',
                      'compute_integrals',
                      'concatenate'])

backend.name = 'torch'
backend.cdgmm3d = cdgmm3d
backend.fft = fft
backend.concatenate = concatenate
backend.modulus = complex_modulus
backend.modulus_rotation = modulus_rotation
backend.compute_integrals = compute_integrals
