"""Author: Louis Thiry, All rights reserved, 2018."""
from collections import defaultdict

import torch
import numpy as np
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



def compute_integrals(input_array, integral_powers):
    """
    Computes integrals of the input_array to the given powers.
        \\int input_array^p
    
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



