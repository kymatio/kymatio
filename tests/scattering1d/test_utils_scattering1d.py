import math
import numpy as np
import pytest
from kymatio import Scattering1D
from kymatio.scattering1d.frontend.numpy_frontend import ScatteringNumPy1D
from kymatio.scattering1d.filter_bank import (compute_params_filterbank,
    get_max_dyadic_subsampling)
from kymatio.scattering1d.utils import (compute_border_indices, compute_padding)


def test_compute_padding():
    """
    Test the compute_padding function
    """

    pad_left, pad_right = compute_padding(32, 16)
    assert pad_left == 8 and pad_right == 8

    with pytest.raises(ValueError) as ve:
        _, _ = compute_padding(8, 16)
    assert "should be larger" in ve.value.args[0]

    with pytest.raises(ValueError) as ve:
        _, _ = compute_padding(64, 16)
    assert "Too large padding value" in ve.value.args[0]


def test_border_indices(random_state=42):
    """
    Tests whether the border indices to unpad are well computed
    """
    rng = np.random.RandomState(random_state)
    J_signal = 10  # signal lives in 2**J_signal
    J = 6  # maximal subsampling

    T = 2**J_signal

    i0 = rng.randint(0, T // 2 + 1, 1)[0]
    i1 = rng.randint(i0 + 1, T, 1)[0]

    x = np.ones(T)
    x[i0:i1] = 0.

    ind_start, ind_end = compute_border_indices(J_signal, J, i0, i1)

    for j in range(J + 1):
        assert j in ind_start.keys()
        assert j in ind_end.keys()
        x_sub = x[::2**j]
        # check that we did take the strict interior
        assert np.max(x_sub[ind_start[j]:ind_end[j]]) == 0.
        # check that we have not forgotten points
        if ind_start[j] > 0:
            assert np.min(x_sub[:ind_start[j]]) > 0.
        if ind_end[j] < x_sub.shape[-1]:
            assert np.min(x_sub[ind_end[j]:]) > 0.


# Check that the default frontend is numpy and that errors are correctly launched.
def test_scattering1d_frontend():
    scattering = Scattering1D(2, shape=(10, ))
    assert isinstance(scattering, ScatteringNumPy1D), 'could not be correctly imported'

    with pytest.raises(RuntimeError) as ve:
        scattering = Scattering1D(2, shape=(10,), frontend='doesnotexist')
    assert "is not valid" in ve.value.args[0]

# Check that self.meta() is compliant with the legacy implementation (see below)
def test_meta():
    J = 6
    Q = (3, 1)
    length = 1024
    shape = (length,)
    S = Scattering1D(J, shape, Q=Q)
    self_meta = S.meta()
    legacy_meta = legacy_compute_meta_scattering(S.J, S.Q, S.max_order,
        S.r_psi, S.sigma0, S.alpha)


def legacy_compute_meta_scattering(J, Q, max_order, r_psi, sigma0, alpha):
    sigma_min = sigma0 / math.pow(2, J)
    Q1, Q2 = Q
    xi1s, sigma1s = compute_params_filterbank(sigma_min, Q1, r_psi)
    j1s = [get_max_dyadic_subsampling(xi1, sigma1, alpha)
        for xi1, sigma1 in zip(xi1s, sigma1s)]
    xi2s, sigma2s = compute_params_filterbank(sigma_min, Q2, r_psi)
    j2s = [get_max_dyadic_subsampling(xi2, sigma2, alpha)
        for xi2, sigma2 in zip(xi2s, sigma2s)]

    meta = {}

    meta['order'] = [[], [], []]
    meta['xi'] = [[], [], []]
    meta['sigma'] = [[], [], []]
    meta['j'] = [[], [], []]
    meta['n'] = [[], [], []]
    meta['key'] = [[], [], []]

    meta['order'][0].append(0)
    meta['xi'][0].append(())
    meta['sigma'][0].append(())
    meta['j'][0].append(())
    meta['n'][0].append(())
    meta['key'][0].append(())

    for (n1, (xi1, sigma1, j1)) in enumerate(zip(xi1s, sigma1s, j1s)):
        meta['order'][1].append(1)
        meta['xi'][1].append((xi1,))
        meta['sigma'][1].append((sigma1,))
        meta['j'][1].append((j1,))
        meta['n'][1].append((n1,))
        meta['key'][1].append((n1,))

        if max_order < 2:
            continue

        for (n2, (xi2, sigma2, j2)) in enumerate(zip(xi2s, sigma2s, j2s)):
            if j2 > j1:
                meta['order'][2].append(2)
                meta['xi'][2].append((xi1, xi2))
                meta['sigma'][2].append((sigma1, sigma2))
                meta['j'][2].append((j1, j2))
                meta['n'][2].append((n1, n2))
                meta['key'][2].append((n1, n2))

    for field, value in meta.items():
        meta[field] = value[0] + value[1] + value[2]

    pad_fields = ['xi', 'sigma', 'j', 'n']
    pad_len = max_order

    for field in pad_fields:
        meta[field] = [x + (math.nan,) * (pad_len - len(x)) for x in meta[field]]

    array_fields = ['order', 'xi', 'sigma', 'j', 'n']

    for field in array_fields:
        meta[field] = np.array(meta[field])

    return meta
