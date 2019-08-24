import torch
import warnings

BACKEND_NAME = 'torch'
from collections import namedtuple


def iscomplex(input):
    return input.size(-1) == 2


def to_complex(input):
    output = input.new(input.size() + (2,)).fill_(0)
    output[..., 0] = input
    return output

def complex_modulus(input_array):
    modulus = torch.zeros_like(input_array)
    modulus[..., 0] += torch.sqrt((input_array ** 2).sum(-1))
    return modulus



def fft(input, inverse=False):
    """
        fft of a 3d signal

        Example
        -------
        x = torch.randn(128, 32, 32, 32, 2)
        x_fft = fft(x, inverse=True)

        Parameters
        ----------
        input : tensor
            complex input for the FFT
        inverse : bool
            True for computing the inverse FFT.
.
    """
    if not iscomplex(input):
        raise(TypeError('The input should be complex (e.g. last dimension is 2)'))
    if inverse:
        return torch.ifft(input, 3)
    return torch.fft(input, 3)


def cdgmm3d(A, B):
    """
    Pointwise multiplication of complex tensors.

    ----------
    A: complex torch tensor
    B: complex torch tensor of the same size as A

    Returns
    -------
    output : torch tensor of the same size as A containing the result of the 
             elementwise complex multiplication of  A with B 
    """
    if not A.is_contiguous():
        warnings.warn("cdgmm3d: tensor A is converted to a contiguous array")
        A = A.contiguous()
    if not B.is_contiguous():
        warnings.warn("cdgmm3d: tensor B is converted to a contiguous array")
        B = B.contiguous()

    if A.size()[-4:] != B.size():
        raise RuntimeError(
            'The tensors are not compatible for multiplication!')

    if not iscomplex(A) or not iscomplex(B):
        raise TypeError('The input, filter and output should be complex')

    if B.ndimension() != 4:
        raise RuntimeError('The second tensor must be simply a complex array!')

    if type(A) is not type(B):
        raise RuntimeError('A and B should be same type!')

    C = torch.empty_like(A)

    C[..., 0] = A[..., 0] * B[..., 0] - A[..., 1] * B[..., 1]
    C[..., 1] = A[..., 0] * B[..., 1] + A[..., 1] * B[..., 0]

    return C

def finalize(s_order_1, s_order_2, max_order):
    if max_order == 2:
        return torch.cat([torch.stack(s_order_1, dim=2), torch.stack(s_order_2, dim=2)], 1)
    else:
        return torch.stack(s_order_1, dim=2)



def modulus_rotation(x, module):
    """
    Computes the convolution with a set of solid harmonics of scale j and
    degree l and returns the square root of their squared sum over m

    Parameters
    ----------
    input_array : tensor
        size (batchsize, M, N, O, 2)
    l : int
        solid harmonic degree l

    j : int
        solid harmonic scale j

    Returns
    -------

    output : torch tensor
        tensor of the same size as input_array. It holds the output of
        the operation::

        $\\sqrt{\\sum_m (\\text{input}_\\text{array} \\star \\psi_{j,l,m})^2)}$

        which is covariant to 3D translations and rotations

    """
    if module == None:
        module = torch.zeros_like(x)
    else:
        module = module **2
    module[..., 0] += (x**2).sum(-1)
    return torch.sqrt(module)




def _low_pass_filter(input_array, low_pass):
    """
    Computes the convolution of input_array with a lowpass filter phi_j

    Parameters
    ----------
    input_array : tensor
        size (batchsize, M, N, O, 2)

    j: int

    Returns
    -------
    output: the result of input_array :math:`\\star phi_J`

    """
    #low_pass = self.gaussian_filters[j]
    return _fft_convolve(input_array, low_pass)


def _compute_standard_scattering_coefs(input_array, low_pass, J, subsample):
    """
    Computes the convolution of input_array with a lowpass filter phi_J
    and downsamples by a factor J.

    Parameters
    ----------
    input_array: torch tensor of size (batchsize, M, N, O, 2)

    Returns
    -------
    output: the result of input_array \\star phi_J downsampled by a factor J

    """
    convolved_input = _low_pass_filter(input_array, low_pass)
    return subsample(convolved_input, J)



def _compute_local_scattering_coefs(input_array, low_pass, points):
    """
    Computes the convolution of input_array with a lowpass filter phi_j and
    and returns the value of the output at particular points

    Parameters
    ----------
    input_array: torch tensor
        size (batchsize, M, N, O, 2)
    points: torch tensor
        size (batchsize, number of points, 3)
    j: int
        the lowpass scale j of phi_j

    Returns
    -------
    output: torch tensor of size (batchsize, number of points, 1) with
            the values of the lowpass filtered moduli at the points given.

    """
    local_coefs = torch.zeros(input_array.size(0), points.size(1), 1)
    convolved_input = _low_pass_filter(input_array, low_pass)#j + 1)
    for i in range(input_array.size(0)):
        for j in range(points[i].size(0)):
            x, y, z = points[i, j, 0], points[i, j, 1], points[i, j, 2]
            local_coefs[i, j, 0] = convolved_input[
                i, int(x), int(y), int(z), 0]
    return local_coefs




def averaging(input_array, method, args, filter, compute_integrals):
    """
    Computes the scattering coefficients out with any of the three methods
    'standard', 'local', 'integral'

    Parameters
    ----------
    input_array : torch tensor
        size (batchsize, M, N, O, 2)
    method : string
        method name with three possible values ("standard", "local", "integral")
    args : dict
        method specific arguments. It methods is equal to "standard", then one
        expects the array args['integral_powers'] to be a list that holds
        the exponents the moduli. It should be raised to before calculating
        the integrals. If method is equal to "local", args['points'] must contain
        a torch tensor of size (batchsize, number of points, 3) the points in
        coordinate space at which you want the moduli sampled
    j : int
        lowpass scale j of :math:`\\phi_j`

    Returns
    -------
    output: torch tensor
        The scattering coefficients as given by different methods.

    """
    methods = ['standard', 'local', 'integral']
    if (not method in methods):
        raise (ValueError('method must be in {}'.format(methods)))
    if method == 'integral':
        return compute_integrals(input_array[..., 0],
                                 args['integral_powers'])
    elif method == 'local':
        return _compute_local_scattering_coefs(input_array, args['points'], filter)
    elif method == 'standard':
        return _compute_standard_scattering_coefs(input_array)


def subsample(input_array, j):
    return input_array[..., ::2 ** j, ::2 ** j, ::2 ** j, :].contiguous()



def compute_integrals(input_array, integral_powers):
    """
        Computes integrals of the input_array to the given powers.

        Parameters
        ----------
        input_array: torch tensor
            size (B, M, N, O), B batch_size, M, N, O spatial dims

        integral_powers: list
            list of P positive floats containing the p values used to
            compute the integrals of the input_array to the power p (l_p norms)

        Returns
        -------
        integrals: torch tensor
            tensor of size (B, P) containing the integrals of the input_array
            to the powers p (l_p norms)

    """
    integrals = torch.zeros(input_array.size(0), len(integral_powers), 1)
    for i_q, q in enumerate(integral_powers):
        integrals[:, i_q, 0] = (input_array ** q).view(
                                        input_array.size(0), -1).sum(1).cpu()
    return integrals




backend = namedtuple('backend', ['name', 'cdgmm3d', 'fft', 'Pad', 'finalize',
    'modulus', 'modulus_rotation', 'subsample', 'compute_integrals'])

backend.name = 'torch'
backend.cdgmm3d = cdgmm3d
backend.fft = fft
backend.finalize = finalize
backend.modulus = complex_modulus
backend.modulus_rotation = modulus_rotation()
backend.subsample = subsample
backend.compute_integrals = compute_integrals
