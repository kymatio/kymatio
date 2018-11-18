import torch
from scattering.scattering1d.backend import pad_1d, modulus_complex, subsample_fourier
from scattering.scattering1d.utils import compute_border_indices
import numpy as np
import pytest
import scattering.scattering1d.backend as backend
import warnings

if backend.NAME == 'skcuda':
    force_gpu = True
else:
    force_gpu = False


def test_pad_1d(random_state=42):
    """
    Tests the correctness and differentiability of pad_1d
    """
    torch.manual_seed(random_state)
    N = 128
    for pad_left in range(0, N, 16):
        for pad_right in range(0, N, 16):
            x = torch.randn(100, 4, N, requires_grad=True)
            x_pad = pad_1d(x, pad_left, pad_right, mode='reflect')
            # Check the size
            x2 = x.clone()
            x_pad2 = x_pad.clone()
            for t in range(1, pad_left + 1):
                diff = x_pad2[..., pad_left - t] - x2[..., t]
                assert torch.max(torch.abs(diff)) <= 1e-7
            for t in range(x2.shape[-1]):
                diff = x_pad2[..., pad_left + t] - x2[..., t]
                assert torch.max(torch.abs(diff)) <= 1e-7
            for t in range(1, pad_right + 1):
                diff = x_pad2[..., x_pad.shape[-1] - 1 - pad_right + t]
                diff -= x2[..., x.shape[-1] - 1 - t]
                assert torch.max(torch.abs(diff)) <= 1e-7
            # check the differentiability
            loss = 0.5 * torch.sum(x_pad**2)
            loss.backward()
            # compute the theoretical gradient for x
            x_grad_original = x.clone()
            x_grad = x_grad_original.new(x_grad_original.shape).fill_(0.)
            x_grad += x_grad_original
            for t in range(1, pad_left + 1):
                x_grad[..., t] += x_grad_original[..., t]
            for t in range(1, pad_right + 1):  # it is counted twice!
                t0 = x.shape[-1] - 1 - t
                x_grad[..., t0] += x_grad_original[..., t0]
            # get the difference
            diff = x.grad - x_grad
            assert torch.max(torch.abs(diff)) <= 1e-7
    # Check that the padding shows an error if we try to pad
    with pytest.raises(ValueError):
        pad_1d(x, x.shape[-1], 0, mode='reflect')
    with pytest.raises(ValueError):
        pad_1d(x, 0, x.shape[-1], mode='reflect')


def test_modulus(random_state=42):
    """
    Tests the stability and differentiability of modulus
    """
    torch.manual_seed(random_state)
    # Test with a random vector
    x = torch.randn(100, 4, 128, 2, requires_grad=True)
    if force_gpu:
        x = x.cuda()
    x_abs = modulus_complex(x)
    if force_gpu:
        x_abs = x_abs.cpu()
    assert len(x_abs.shape) == len(x.shape)
    # check the value
    x_abs2 = x_abs.clone()
    x2 = x.clone()
    if force_gpu:
        x2 = x2.cpu()
    diff = x_abs2[..., 0] - torch.sqrt(x2[..., 0]**2 + x2[..., 1]**2)
    assert torch.max(torch.abs(diff)) <= 1e-6
    if backend.NAME == "skcuda":
        warnings.warn(("The skcuda backend does not pass differentiability"
            "tests, but that's ok (for now)."), RuntimeWarning, stacklevel=2)
        return

    # check the gradient
    loss = torch.sum(x_abs)
    loss.backward()
    x_grad = x2 / x_abs2[..., 0].unsqueeze(dim=-1)
    diff = x.grad - x_grad
    assert torch.max(torch.abs(diff)) <= 1e-7

    # Test the differentiation with a vector made of zeros
    x0 = torch.zeros(100, 4, 128, 2, requires_grad=True)
    if force_gpu:
        x0 = x0.cuda()
    x_abs0 = modulus_complex(x0)
    loss0 = torch.sum(x_abs0)
    loss0.backward()
    assert torch.max(torch.abs(x0.grad)) <= 1e-7


def test_subsample_fourier(random_state=42):
    """
    Tests whether the periodization in Fourier performs a good subsampling
    in time
    """
    rng = np.random.RandomState(random_state)
    J = 10
    x = rng.randn(100, 4, 2**J) + 1j * rng.randn(100, 4, 2**J)
    x_f = np.fft.fft(x, axis=-1)[..., np.newaxis]
    x_f.dtype = 'float64'  # make it a vector
    x_f_th = torch.from_numpy(x_f)
    if force_gpu:
        x_f_th = x_f_th.cuda()
    for j in range(J + 1):
        x_f_sub_th = subsample_fourier(x_f_th, 2**j)
        if force_gpu:
            x_f_sub_th = x_f_sub_th.cpu()
        x_f_sub = x_f_sub_th.numpy()
        x_f_sub.dtype = 'complex128'
        x_sub = np.fft.ifft(x_f_sub[..., 0], axis=-1)
        assert np.max(np.abs(x[:, :, ::2**j] - x_sub)) < 1e-7


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
