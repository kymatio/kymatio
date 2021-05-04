import pytest
import numpy as np
import torch
from kymatio.scattering1d.backend.agnostic_backend import pad

run_without_pytest = 1


def _test_padding(backend_name):
    if backend_name == 'numpy':
        backend = np
    elif backend_name == 'torch':
        backend = torch

    for N in (128, 129):  # even, odd
        x = backend.arange(6 * N).reshape(2, 3, N)
        for pad_factor in (1, 2, 3, 4):
            pad_left = (N // 2) * pad_factor
            pad_right = int(np.ceil(N / 4) * pad_factor)

            for pad_mode in ('zero', 'reflect'):
                out0 = pad(x, pad_left, pad_right, pad_mode=pad_mode,
                           backend_name=backend_name)
                out1 = np.pad(x,
                              [[0, 0]] * (x.ndim - 1) + [[pad_left, pad_right]],
                              mode=pad_mode if pad_mode != 'zero' else 'constant')

                if backend_name == 'torch':
                    out1 = torch.from_numpy(out1)
                assert backend.allclose(out0, out1), (
                    "{} | (N, pad_mode, pad_left, pad_right) = ({}, {}, {}, {})"
                    ).format(backend_name, N, pad_mode, pad_left, pad_right)


def test_numpy():
    _test_padding('numpy')


def test_torch():
    _test_padding('torch')


if __name__ == '__main__':
    if run_without_pytest:
        test_numpy()
        test_torch()
    else:
        pytest.main([__file__, "-s"])
