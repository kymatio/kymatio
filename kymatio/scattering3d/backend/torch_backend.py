import torch
import warnings

BACKEND_NAME = 'torch'
from collections import namedtuple


def _iscomplex(input):
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


def _compute_standard_scattering_coefs(input_array, filter_list, J, subsample):
    """Computes convolution and downsamples.

        Computes the convolution of input_array with a lowpass filter phi_J
        and downsamples by a factor J.

        Parameters
        ----------
        input_array : torch Tensor
            Size (batchsize, M, N, O, 2).
        filter_list : list of torch Tensors
            Size (M, N, O, 2).
        J : int
            Low pass scale of phi_J.
        subsample : function
            Subsampling function.

        Returns
        -------
        output : tensor
            The result of input_array \\star phi_J downsampled by a factor J.

    """
    low_pass = filter_list[J]
    convolved_input = cdgmm3d(input_array, low_pass)
    convolved_input = fft(convolved_input, inverse=True)
    return subsample(convolved_input, J)


def _compute_local_scattering_coefs(input_array, filter_list, j, points):
    """Compute convolution and returns particular points.

        Computes the convolution of input_array with a lowpass filter phi_j+1
        and returns the value of the output at particular points.
        Parameters
        ----------
        input_array : torch tensor
            Size (batchsize, M, N, O, 2).
        filter_list : list of torch Tensors
            Size (M, N, O, 2).
        j : int
            The lowpass scale j of phi_j.
        points : torch tensor
            Size (batchsize, number of points, 3).
        Returns
        -------
        output : torch tensor
            Torch tensor of size (batchsize, number of points, 1) with the values
            of the lowpass filtered moduli at the points given.
    """
    local_coefs = torch.zeros(input_array.shape[0], points.shape[1], 1)
    low_pass = filter_list[j + 1]
    convolved_input = cdgmm3d(input_array, low_pass)
    convolved_input = fft(convolved_input, inverse=True)
    for i in range(input_array.shape[0]):
        for j in range(points[i].shape[0]):
            x, y, z = points[i, j, 0], points[i, j, 1], points[i, j, 2]
            local_coefs[i, j, 0] = convolved_input[
                i, int(x), int(y), int(z), 0]
    return local_coefs


def subsample(input_array, j):
    """Downsamples.
        Parameters
        ----------
        input_array : tensor
            Input tensor of shape (batch, channel, M, N, O, 2).
        j : int
            Downsampling factor.
        Returns
        -------
        out : tensor
            Downsampled tensor of shape (batch, channel, M // 2 ** j, N // 2
            ** j, O // 2 ** j, 2).
    """
    return input_array[..., ::2 ** j, ::2 ** j, ::2 ** j, :].contiguous()


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
    integrals = torch.zeros(input_array.shape[0], len(integral_powers), 1)
    for i_q, q in enumerate(integral_powers):
        integrals[:, i_q, 0] = (input_array ** q).view(
            input_array.shape[0], -1).sum(1).cpu()
    return integrals


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
    if not _iscomplex(input):
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

    if not _iscomplex(A) or not _iscomplex(B):
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
                      'subsample',
                      'compute_integrals',
                      'concatenate'])

backend.name = 'torch'
backend.cdgmm3d = cdgmm3d
backend.fft = fft
backend.concatenate = concatenate
backend.modulus = complex_modulus
backend.modulus_rotation = modulus_rotation
backend.subsample = subsample
backend.compute_integrals = compute_integrals
backend._compute_standard_scattering_coefs = _compute_standard_scattering_coefs
backend._compute_local_scattering_coefs = _compute_local_scattering_coefs
