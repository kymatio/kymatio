import pytest
import numpy as np
import warnings
from kymatio.scattering1d.backend.agnostic_backend import pad

run_without_pytest = 0


def _test_padding(backend_name):
    """Test that agnostic implementation matches numpy's."""
    def _arange(N):
        if backend_name == 'tensorflow':
            return backend.range(N)
        return backend.arange(N)

    backend = _get_backend(backend_name)

    for N in (128, 129):  # even, odd
        x = backend.reshape(_arange(6 * N), (2, 3, N))
        for pad_factor in (1, 2, 3, 4):
            pad_left = (N // 2) * pad_factor
            pad_right = int(np.ceil(N / 4) * pad_factor)

            for pad_mode in ('zero', 'reflect'):
                out0 = pad(x, pad_left, pad_right, pad_mode=pad_mode)
                out1 = np.pad(x,
                              [[0, 0]] * (x.ndim - 1) + [[pad_left, pad_right]],
                              mode=pad_mode if pad_mode != 'zero' else 'constant')

                out0 = out0.numpy() if hasattr(out0, 'numpy') else out0
                assert np.allclose(out0, out1), (
                    "{} | (N, pad_mode, pad_left, pad_right) = ({}, {}, {}, {})"
                    ).format(backend_name, N, pad_mode, pad_left, pad_right)

def _test_pad_axis(backend_name):
    """Test that padding any N-dim axis works as expected."""
    backend = _get_backend(backend_name)
    x = backend.zeros((5, 6, 7, 8, 9, 10, 11))

    pad_left, pad_right = 4, 5
    kw = dict(pad_left=pad_left, pad_right=pad_right, pad_mode='reflect')

    for axis in range(x.ndim):
        if backend_name == 'tensorflow' and axis != x.ndim - 1:
            # implemented only for last axis
            continue
        shape0 = list(x.shape)
        shape0[axis] += (pad_left + pad_right)
        shape1 = pad(x, axis=axis, **kw).shape
        shape2 = pad(x, axis=axis - x.ndim, **kw).shape  # negative axis version

        assert np.allclose(shape0, shape1)
        assert np.allclose(shape0, shape2)

def test_pad_numpy():
    _test_padding('numpy')
    _test_pad_axis('numpy')


def test_pad_torch():
    if cant_import('torch'):
        return
    _test_padding('torch')
    _test_pad_axis('torch')


def test_pad_tensorflow():
    if cant_import('tensorflow'):
        return
    _test_padding('tensorflow')
    _test_pad_axis('tensorflow')

#### utilities ###############################################################
def _get_backend(backend_name):
    if backend_name == 'numpy':
        backend = np
    elif backend_name == 'torch':
        import torch
        backend = torch
    elif backend_name == 'tensorflow':
        import tensorflow as tf
        backend = tf
    return backend

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

if __name__ == '__main__':
    if run_without_pytest:
        test_pad_numpy()
        test_pad_torch()
        test_pad_tensorflow()
    else:
        pytest.main([__file__, "-s"])
