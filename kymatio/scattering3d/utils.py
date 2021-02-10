"""Author: Louis Thiry"""
import numpy as np
import warnings


def generate_weighted_sum_of_gaussians(grid, positions, weights, sigma):
    """
        Computes sum of 3D Gaussians centered at given positions and weighted
        with the given weights.
        Parameters
        ----------
        grid : numpy array
            numerical grid, size (3, M, N, O)
        positions: numpy array
            positions of the Gaussians, size (B, N_gaussians, 3)
            B batch_size, N_gaussians number or gaussians
        weights: numpy array
            weights of the Gaussians, size (B, N_gaussians)
            zero weights are assumed to be at the end since if a weight is zero
            all weights after are ignored
        sigma : float
            width parameter of the Gaussian
        Returns
        -------
        signals : numpy array
            numpy array of size (B, M, N, O)
            B is the batch_size, M, N, O are the size of the signal
    """
    _, M, N, O = grid.shape
    signals = np.zeros((positions.shape[0], M, N, O))

    for i_signal in range(positions.shape[0]):
        n_points = positions[i_signal].shape[0]
        for i_point in range(n_points):
            if weights[i_signal, i_point] == 0:
                break
            weight = weights[i_signal, i_point]
            center = positions[i_signal, i_point]
            signals[i_signal] = np.add(signals[i_signal] , weight * np.exp(
                -0.5 * ((grid[0] - center[0]) ** 2 +
                        (grid[1] - center[1]) ** 2 +
                        (grid[2] - center[2]) ** 2) / sigma**2))
    return signals / ((2 * np.pi) ** 1.5 * sigma ** 3)

def get_3d_angles(cartesian_grid):
    """
        Given a cartesian grid, computes the spherical coord angles (theta, phi).
        Parameters
        ----------
        cartesian_grid: numpy array
            4D array of shape (3, M, N, O)
        Returns
        -------
        polar: numpy array
            polar angles, shape (M, N, O)
        azimutal: numpy array
            azimutal angles, shape (M, N, O)
    """
    z, y, x = cartesian_grid
    azimuthal = np.arctan2(y, x)
    rxy = sqrt(x ** 2 + y ** 2)
    polar = np.arctan2(z, rxy) + np.pi / 2
    return polar, azimuthal


def double_factorial(i):
    """Computes the double factorial of an integer."""
    return 1 if (i < 1) else np.prod(np.arange(i, 0, -2))


def sqrt(x):
    """
        Compute the square root of an array
        This suppresses any warnings due to invalid input, unless the array is
        real and has negative values. This fixes the erroneous warnings
        introduced by an Intel SVM bug for large single-precision arrays. For
        more information, see:
            https://github.com/numpy/numpy/issues/11448
            https://github.com/ContinuumIO/anaconda-issues/issues/9129
        Parameters
        ----------
        x : numpy array
            An array for which we would like to compute the square root.
        Returns
        -------
        y : numpy array
            The square root of the array.
    """
    if np.isrealobj(x) and (x < 0).any():
        warnings.warn("Negative value encountered in sqrt", RuntimeWarning,
            stacklevel=1)
    old_settings = np.seterr(invalid='ignore')
    y = np.sqrt(x)
    np.seterr(**old_settings)

    return y

def _apply_filters(filters, fn):
    """
        Parameters
        ----------
        filters: a filter bank
        fn: a function to apply on the parameters
        Returns
        -------
        filters: the filters modified
    """
    for k in range(len(filters)):
        filters[k] = fn(filters[k])
    return filters
