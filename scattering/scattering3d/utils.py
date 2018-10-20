"""Author: Louis Thiry, All rights reserved, 2018."""
from collections import defaultdict

import torch
from skcuda import cufft
import numpy as np
try:
    import pyfftw
except:
    pass
    # XXX This is try/excepted in order to not hard require pyfftw installed
    # This can be turned into a scipy fftpack import
    # But the current state of this file requires substantial
    # modification of the CPU functionality, because scipy fftpack
    # does not store plans and would have to function in a different way.
    # To be seen whether we might be able to get away with pytorch CPU fft

import warnings


def generate_weighted_sum_of_diracs(positions, weights, M, N, O, 
                                    sigma_dirac=0.4):
    n_signals = positions.shape[0]
    signals = torch.zeros(n_signals, M, N, O)
    d_s = [(0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1),
           (1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)]
    values = torch.FloatTensor(8)

    for i_signal in range(n_signals):
        n_positions = positions[i_signal].shape[0]
        for i_position in range(n_positions):
            position = positions[i_signal, i_position]
            i, j, k = torch.floor(position).type(torch.IntTensor)
            for i_d, (d_i, d_j, d_k) in enumerate(d_s):
                values[i_d] = np.exp(-0.5 * (
                    (position[0] - (i + d_i)) ** 2 + 
                    (position[1] - (j + d_j)) ** 2 + 
                    (position[2] - (k + d_k)) ** 2) / sigma_dirac ** 2)
            values *= weights[i_signal, i_position] / values.sum()
            for i_d, (d_i, d_j, d_k) in enumerate(d_s):
                i_, j_, k_ = (i + d_i) % M, (j + d_j) % N, (k + d_k) % O
                signals[i_signal, i_, j_, k_] += values[i_d]

    return signals


def generate_large_weighted_sum_of_gaussians(positions, weights, M, N, O, 
                                             fourier_gaussian, fft=None):
    n_signals = positions.shape[0]
    signals = torch.zeros(n_signals, M, N, O, 2)
    signals[..., 0] = generate_weighted_sum_of_diracs(
        positions, weights, M, N, O)

    if fft is None:
        fft = Fft3d()
    return fft(cdgmm3d(fft(signals, inverse=False), fourier_gaussian),
               inverse=True, normalized=True)[..., 0]


def generate_weighted_sum_of_gaussians(grid, positions, weights, sigma,
                                       cuda=False):
    _, M, N, O = grid.size()
    signals = torch.zeros(positions.size(0), M, N, O)
    if cuda:
        signals = signals.cuda()

    for i_signal in range(positions.size(0)):
        n_points = positions[i_signal].size(0)
        for i_point in range(n_points):
            if weights[i_signal, i_point] == 0:
                break
            weight = weights[i_signal, i_point]
            center = positions[i_signal, i_point]
            signals[i_signal] += weight * torch.exp(
                -0.5 * ((grid[0] - center[0]) ** 2 + 
                        (grid[1] - center[1]) ** 2 + 
                        (grid[2] - center[2]) ** 2) / sigma**2)
    return signals / ((2 * np.pi) ** 1.5 * sigma ** 3)


def subsample(input_array, j):
    return input_array.unfold(3, 1, 2 ** j
                     ).unfold(2, 1, 2 ** j
                     ).unfold(1, 1, 2 ** j).contiguous()


def complex_modulus(input_array):
    modulus = torch.zeros_like(input_array)
    modulus[..., 0] += torch.sqrt((input_array ** 2).sum(-1))
    return modulus


def compute_integrals(input_array, integral_powers):
    """
    Computes integrals of the input_array to the given powers.
        \int input_array^p
    
    Parameters
    ----------
    input_array: torch tensor 
    
    integral_powers: list
        list that contains the powers p

    Returns
    -------
    output: torch tensor
        the integrals of the powers of the input_array    
    """
    integrals = torch.zeros(input_array.size(0), len(integral_powers), 1)
    for i_q, q in enumerate(integral_powers):
        integrals[:, i_q, 0] = (input_array ** q).view(
                                        input_array.size(0), -1).sum(1).cpu()
    return integrals


def get_3d_angles(cartesian_grid):
    """
    Given a cartesian grid, computes the spherical coord angles (theta, phi).

    Parameters
    ----------
    cartesian_grid: torch tensor
                  4D tensor of shape (3, M, N, O)
    Returns
    -------
    output: tuple
        tuple of two elements. The first is the polar coordinates of the grid 
        and the second the azimuthal coordinates.
        Both of them are 3D tensors of shape (M, N, O).
    """
    z, y, x = cartesian_grid
    azimuthal = np.arctan2(y, x)
    rxy = np.sqrt(x ** 2 + y ** 2)
    polar = np.arctan2(z, rxy) + np.pi / 2
    return polar, azimuthal


def double_factorial(l):
    return 1 if (l < 1) else np.prod(np.arange(l, 0, -2))


def getDtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


def iscomplex(input):
    return input.size(-1) == 2


def to_complex(input):
    output = input.new(input.size() + (2,)).fill_(0)
    output[..., 0] = input
    return output


class Fft3d(object):
    """This class builds a wrapper to 3D FFTW on CPU / cuFFT on nvidia GPU."""

    def __init__(self, n_fftw_threads=8):
        self.n_fftw_threads = n_fftw_threads
        self.fftw_cache = defaultdict(lambda: None)
        self.cufft_cache = defaultdict(lambda: None)

    def buildCufftCache(self, input, type):
        batch_size, M, N, O, _ = input.size()
        signal_dims = np.asarray([M, N, O], np.int32)
        batch = batch_size
        idist = M * N * O
        istride = 1
        ostride = istride
        odist = idist
        rank = 3
        plan = cufft.cufftPlanMany(rank, signal_dims.ctypes.data,
                                         signal_dims.ctypes.data,
                                   istride, idist, signal_dims.ctypes.data, 
                                   ostride, odist, type, batch)
        self.cufft_cache[(input.size(), type, input.get_device())] = plan

    def buildFftwCache(self, input, inverse):
        direction = 'FFTW_BACKWARD' if inverse else 'FFTW_FORWARD'
        batch_size, M, N, O, _ = input.size()
        fftw_input_array = pyfftw.empty_aligned(
            (batch_size, M, N, O), dtype='complex64')
        fftw_output_array = pyfftw.empty_aligned(
            (batch_size, M, N, O), dtype='complex64')
        fftw_object = pyfftw.FFTW(fftw_input_array, fftw_output_array,
                                  axes=(1, 2, 3), direction=direction,
                                  threads=self.n_fftw_threads)
        self.fftw_cache[(input.size(), inverse)] = (
            fftw_input_array, fftw_output_array, fftw_object)

    def __call__(self, input, inverse=False, normalized=False):
        if not isinstance(input, torch.cuda.FloatTensor):
            if not isinstance(input, (torch.FloatTensor, torch.DoubleTensor)):
                raise(TypeError('The input should be a torch.cuda.FloatTensor, \
                                torch.FloatTensor or a torch.DoubleTensor'))
            else:
                def f(x): return np.stack([x.real, x.imag], axis=len(x.shape))
                if(self.fftw_cache[(input.size(), inverse)] is None):
                    self.buildFftwCache(input, inverse)
                input_arr, output_arr, fftw_obj = self.fftw_cache[(
                    input.size(), inverse)]

                input_arr.real[:] = input[..., 0]
                input_arr.imag[:] = input[..., 1]
                fftw_obj()

                return torch.from_numpy(f(output_arr))

        if not input.is_contiguous():
            raise(RuntimeError("input is not contiguous"))

        output = input.new(input.size())
        flag = cufft.CUFFT_INVERSE if inverse else cufft.CUFFT_FORWARD
        ffttype = cufft.CUFFT_C2C if isinstance(
            input, torch.cuda.FloatTensor) else cufft.CUFFT_Z2Z
        if self.cufft_cache[
                    (input.size(), ffttype, input.get_device())] is None:
            self.buildCufftCache(input, ffttype)
        cufft.cufftExecC2C(self.cufft_cache[(input.size(), ffttype, 
                    input.get_device())], input.data_ptr(),
                    output.data_ptr(), flag)
        if normalized:
            output /= input.size(1) * input.size(2) * input.size(3)
        return output


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
