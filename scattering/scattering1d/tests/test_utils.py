import torch
from torch.autograd import Variable
from scattering.scattering1d.utils import pad1D, modulus, subsample_fourier
from scattering.scattering1d.utils import compute_border_indices
import numpy as np
from sklearn.utils import check_random_state
import pytest


def test_pad1D(random_state=42):
    """
    Tests the correctness and differentiability of pad1D
    """
    torch.manual_seed(random_state)
    N = 128
    for pad_left in range(0, N, 16):
        for pad_right in range(0, N, 16):
            x = Variable(torch.randn(100, 4, N), requires_grad=True)
            x_pad = pad1D(x, pad_left, pad_right, mode='reflect')
            # Check the size
            x2 = x.data.clone()
            x_pad2 = x_pad.data.clone()
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
            x_grad_original = x.data.clone()
            x_grad = x_grad_original.new(x_grad_original.shape).fill_(0.)
            x_grad += x_grad_original
            for t in range(1, pad_left + 1):
                x_grad[..., t] += x_grad_original[..., t]
            for t in range(1, pad_right + 1):  # it is counted twice!
                t0 = x.shape[-1] - 1 - t
                x_grad[..., t0] += x_grad_original[..., t0]
            # get the difference
            diff = x.grad.data - x_grad
            assert torch.max(torch.abs(diff)) <= 1e-7
    # Check that the padding shows an error if we try to pad
    with pytest.raises(ValueError):
        pad1D(x, x.shape[-1], 0, mode='reflect')
    with pytest.raises(ValueError):
        pad1D(x, 0, x.shape[-1], mode='reflect')


def test_modulus(random_state=42):
    """
    Tests the stability and differentiability of modulus
    """
    torch.manual_seed(random_state)
    # Test with a random vector
    x = Variable(torch.randn(100, 4, 128, 2), requires_grad=True)
    x_abs = modulus(x)
    assert len(x_abs.shape) == len(x.shape) - 1
    # check the value
    x_abs2 = x_abs.data.clone()
    x2 = x.data.clone()
    diff = x_abs2 - torch.sqrt(x2[..., 0]**2 + x2[..., 1]**2)
    assert torch.max(torch.abs(diff)) <= 1e-7
    # check the gradient
    loss = torch.sum(x_abs)
    loss.backward()
    x_grad = x2 / x_abs2.unsqueeze(-1)
    diff = x.grad.data - x_grad
    assert torch.max(torch.abs(diff)) <= 1e-7

    # Test the differentiation with a vector made of zeros
    x0 = Variable(torch.zeros(100, 4, 128, 2), requires_grad=True)
    x_abs0 = modulus(x0)
    loss0 = torch.sum(x_abs0)
    loss0.backward()
    assert torch.max(torch.abs(x0.grad.data)) <= 1e-7


def test_subsample_fourier(random_state=42):
    """
    Tests whether the periodization in Fourier performs a good subsampling
    in time
    """
    rng = check_random_state(random_state)
    J = 10
    x = rng.randn(100, 4, 2**J) + 1j * rng.randn(100, 4, 2**J)
    x_fft = np.fft.fft(x, axis=-1)[..., np.newaxis]
    x_fft.dtype = 'float64'  # make it a vector
    x_fft_th = torch.from_numpy(x_fft)
    for j in range(J + 1):
        x_fft_sub_th = subsample_fourier(x_fft_th, 2**j)
        x_fft_sub = x_fft_sub_th.numpy()
        x_fft_sub.dtype = 'complex128'
        x_sub = np.fft.ifft(x_fft_sub[..., 0], axis=-1)
        assert np.max(np.abs(x[:, :, ::2**j] - x_sub)) < 1e-7


def test_border_indices(random_state=42):
    """
    Tests whether the border indices to unpad are well computed
    """
    rng = check_random_state(random_state)
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
