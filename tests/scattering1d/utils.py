# -*- coding: utf-8 -*-
"""Methods reused in testing."""
import numpy as np
import scipy.signal
import warnings


def cant_import(backend_name):
    if backend_name == 'numpy':
        return False
    elif backend_name == 'torch':
        try:
            import torch
        except ImportError:
            warnings.warn("Failed to import torch")
            return True
    elif backend_name == 'tensorflow':
        try:
            import tensorflow
        except ImportError:
            warnings.warn("Failed to import tensorflow")
            return True


def fdts(N, n_partials=2, total_shift=None, f0=None, seg_len=None,
         endpoint=False):
    """Generate windowed tones with Frequency-dependent Time Shifts (FDTS)."""
    total_shift = total_shift or N//16
    f0 = f0 or N//12
    seg_len = seg_len or N//8

    t = np.linspace(0, 1, N, endpoint=endpoint)
    window = scipy.signal.tukey(seg_len, alpha=0.5)
    pad_right = (N - len(window)) // 2
    pad_left = N - len(window) - pad_right
    window = np.pad(window, [pad_left, pad_right])

    x = np.zeros(N)
    xs = x.copy()
    for p in range(1, 1 + n_partials):
        x_partial = np.sin(2*np.pi * p*f0 * t) * window
        partial_shift = int(total_shift * np.log2(p) / np.log2(n_partials)
                            ) - total_shift//2
        xs_partial = np.roll(x_partial, partial_shift)
        x += x_partial
        xs += xs_partial
    return x, xs


def echirp(N, fmin=.1, fmax=None, tmin=0, tmax=1):
    """https://overlordgolddragon.github.io/test-signals/ (bottom)"""
    fmax = fmax or N // 2
    t = np.linspace(tmin, tmax, N)

    a = (fmin**tmax / fmax**tmin) ** (1/(tmax - tmin))
    b = fmax**(1/tmax) * (1/a)**(1/tmax)

    phi = 2*np.pi * (a/np.log(b)) * (b**t - b**tmin)
    return np.cos(phi)


def _l2(x):
    return np.sqrt(np.sum(np.abs(x)**2))

def l2(x0, x1):
    """Coeff distance measure; Eq 2.24 in
    https://www.di.ens.fr/~mallat/papiers/ScatCPAM.pdf
    """
    return _l2(x1 - x0) / _l2(x0)
