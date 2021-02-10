import numpy as np
import pytest
from kymatio import Scattering1D
from kymatio.scattering1d.frontend.torch_frontend import ScatteringTorch1D
from kymatio.scattering1d.utils import compute_border_indices, compute_padding


def test_compute_padding():
    """
    Test the compute_padding function
    """

    pad_left, pad_right = compute_padding(5, 16)
    assert pad_left == 8 and pad_right == 8

    with pytest.raises(ValueError) as ve:
        _, _ = compute_padding(3, 16)
    assert "should be larger" in ve.value.args[0]

    with pytest.raises(ValueError) as ve:
        _, _ = compute_padding(6, 16)
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

    ind_start, ind_end = compute_border_indices(J, i0, i1)

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
    assert isinstance(scattering, ScatteringTorch1D), 'could not be correctly imported'

    with pytest.raises(RuntimeError) as ve:
        scattering = Scattering1D(2, shape=(10,), frontend='doesnotexist')
    assert "is not valid" in ve.value.args[0]
