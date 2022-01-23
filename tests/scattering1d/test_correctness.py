# -*- coding: utf-8 -*-
"""Tests related to Scattering1D, and for utilities."""
import pytest, warnings
import numpy as np
import scipy.signal
from utils import cant_import
from kymatio import Scattering1D
from kymatio.scattering1d.backend.agnostic_backend import (
    pad, stride_axis, unpad_dyadic, _emulate_get_conjugation_indices)
from kymatio.toolkit import rel_l2

# set True to execute all test functions without pytest
run_without_pytest = 0
# will run most tests with this backend
default_frontend = ('numpy', 'torch', 'tensorflow')[0]


#### Scattering tests ########################################################
def test_T():
    """Test that `T` controls degree of invariance as intended."""
    # configure scattering & signal
    J = 10
    Q = 16
    N = 2048
    width = N//8
    shift = N//8
    T0, T1 = N//2, N//4
    freq_fracs = (4, 8, 16, 32)

    # make signal & shifted
    window = scipy.signal.tukey(width, alpha=0.5)
    window = np.pad(window, (N - width) // 2)
    t  = np.linspace(0, 1, N, endpoint=False)
    x  = np.sum([np.cos(2*np.pi * N/ff * t) for ff in freq_fracs], axis=0
                ) * window
    xs = np.roll(x, shift)

    # make scattering objects
    kw = dict(J=J, Q=Q, shape=N, average=1, out_type="array", pad_mode="zero",
              max_pad_factor=1, frontend=default_frontend)
    ts0 = Scattering1D(T=T0, **kw)
    ts1 = Scattering1D(T=T1, **kw)

    # scatter
    ts0_x  = ts0(x)
    ts0_xs = ts0(xs)
    ts1_x  = ts1(x)
    ts1_xs = ts1(xs)

    # compare distances
    l2_00_xxs = rel_l2(ts0_x, ts0_xs)
    l2_11_xxs = rel_l2(ts1_x, ts1_xs)

    th0, th1 = .021, .15
    assert l2_00_xxs < th0, "{} > {}".format(l2_00_xxs, th0)
    assert l2_11_xxs > th1, "{} < {}".format(l2_11_xxs, th1)


#### Primitives tests ########################################################
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


def _test_subsample_fourier_axis(backend_name):
    """Test that subsampling an arbitrary axis works as expected."""
    backend = _get_backend(backend_name)
    B = _get_kymatio_backend(backend_name)
    x = np.random.randn(4, 8, 16, 32)

    if backend_name == 'torch':
        xb = backend.tensor(x)
    elif backend_name == 'tensorflow':
        xb = backend.cast(backend.convert_to_tensor(x), backend.complex64)
    else:
        xb = x

    k = 2
    axis = 3
    for k in (2, 4):
        for axis in range(x.ndim):
            if (backend_name == 'tensorflow' and
                    axis not in (x.ndim - 1, x.ndim - 2)):
                # not implemented
                continue
            xf = B.fft(xb, axis=axis)
            outf = B.subsample_fourier(xf, k, axis=axis)
            out = B.ifft(outf, axis=axis)

            xref = xb[stride_axis(k, axis, xb.ndim)]
            if backend_name != 'numpy':
                out = out.numpy()
            out = out.real
            assert np.allclose(xref, out, atol=5e-7), np.abs(xref - out).max()


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


def test_subsample_fourier_numpy():
    _test_subsample_fourier_axis('numpy')


def test_subsample_fourier_torch():
    if cant_import('torch'):
        return
    _test_subsample_fourier_axis('torch')


def test_subsample_fourier_tensorflow():
    if cant_import('tensorflow'):
        return
    _test_subsample_fourier_axis('tensorflow')


def test_emulate_get_conjugation_indices():
    """Test that `conj_reflections` indices fetcher works correctly."""
    for N in (123, 124, 125, 126, 127, 128, 129, 159, 180, 2048, 9589, 65432):
      for K in (1, 2, 3, 4, 5, 6):
        for pad_factor in (1, 2, 3, 4):
          for trim_tm in (0, 1, 2, 3, 4):
              if trim_tm > pad_factor:
                  continue
              test_params = dict(N=N, K=K, pad_factor=pad_factor)
              test_params_str = '\n'.join([f'{k}={v}' for k, v in
                                           test_params.items()])

              padded_len = 2**pad_factor * int(2**np.ceil(np.log2(N)))
              pad_left = int(np.ceil((padded_len - N) / 2))
              pad_right = padded_len - pad_left - N

              kw = dict(N=N, K=K, pad_left=pad_left, pad_right=pad_right,
                        trim_tm=trim_tm)
              out0, rp = _get_conjugation_indices(**kw)
              out1 = _emulate_get_conjugation_indices(**kw)

              errmsg = "{}\n{}\n{}".format(out0, out1, test_params_str)

              # first validate the original output ###########################
              # check by sign flipping at obtained slices and adding to
              # original then seeing if it sums to zero with original
              # in at least as many places as there are indices
              rp0 = rp[::2**K]
              rp1 = rp0.copy()
              for slc in out0:
                  rp1[slc] *= -1

              n_zeros_min = sum((s.stop - s.start) for s in out0)
              n_zeros_sum = np.sum((rp0 + rp1) == 0)

              if n_zeros_sum < n_zeros_min:
                  raise AssertionError("{} < {}\n{}".format(
                      n_zeros_sum, n_zeros_min, errmsg))

              # now assert equality with the emulation #######################
              for s0, s1 in zip(out0, out1):
                  assert s0.start == s1.start, errmsg
                  assert s0.stop  == s1.stop,  errmsg


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


def _get_kymatio_backend(backend_name):
    if backend_name == 'numpy':
        from kymatio.scattering1d.backend.numpy_backend import NumpyBackend1D
        return NumpyBackend1D
    elif backend_name == 'torch':
        from kymatio.scattering1d.backend.torch_backend import TorchBackend1D
        return TorchBackend1D
    elif backend_name == 'tensorflow':
        from kymatio.scattering1d.backend.tensorflow_backend import (
            TensorFlowBackend1D)
        return TensorFlowBackend1D


def _get_conjugation_indices(N, K, pad_left, pad_right, trim_tm):
    """Ground truth of the algorithm. Not tested for extreme edge cases, but
    those are impossible in implementation (stride > signal length).
    """
    import numpy as np

    # compute boundary indices from simulated reflected ramp at original length
    r = np.arange(N)
    rp = np.pad(r, [pad_left, pad_right], mode='reflect')
    if trim_tm > 0:
        rp = unpad_dyadic(rp, N, len(rp), len(rp) // 2**trim_tm)

    # will conjugate where sign is negative
    rpdiffo = np.diff(rp)
    rpdiffo = np.hstack([rpdiffo[0], rpdiffo])

    # mark rising ramp as +1, including bounds, and everything else as -1;
    # -1 will conjugate. This instructs to not conjugate: non-reflections, bounds.
    # diff will mark endpoints with sign opposite to rise's; correct manually
    rpdiffo2 = rpdiffo.copy()
    rpdiffo2[np.where(np.diff(rpdiffo) > 0)[0]] = 1

    rpdiff = rpdiffo2[::2**K]

    idxs = np.where(rpdiff == -1)[0]

    # convert to slices ######################################################
    if idxs.size == 0:
        slices_contiguous = []
    else:
        ic = [0, *(np.where(np.diff(idxs) > 1)[0] + 1)]
        ic.append(None)
        slices_contiguous = []
        for i in range(len(ic) - 1):
            s, e = ic[i], ic[i + 1]
            start = idxs[s]
            end = idxs[e - 1] + 1 if e is not None else idxs[-1] + 1
            slices_contiguous.append(slice(start, end))

    out = slices_contiguous
    return out, rp


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
        test_T()
        test_pad_numpy()
        test_pad_torch()
        test_pad_tensorflow()
        test_subsample_fourier_numpy()
        test_subsample_fourier_torch()
        test_subsample_fourier_tensorflow()
        test_emulate_get_conjugation_indices()
    else:
        pytest.main([__file__, "-s"])
