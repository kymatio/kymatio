import pytest
import torch
import numpy as np

backends = []

skcuda_available = False
try:
    if torch.cuda.is_available():
        from skcuda import cublas
        import cupy
        skcuda_available = True
except:
    Warning('torch_skcuda backend not available.')

if skcuda_available:
    from kymatio.scattering1d.backend.torch_skcuda_backend import backend
    backends.append(backend)

from kymatio.scattering1d.backend.torch_backend import backend
backends.append(backend)

if torch.cuda.is_available():
    devices = ['cuda', 'cpu']
else:
    devices = ['cpu']


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_pad_1d(device, backend, random_state=42):
    """
    Tests the correctness and differentiability of pad_1d
    """
    torch.manual_seed(random_state)
    N = 128
    for pad_left in range(0, N - 16, 16):
        for pad_right in [pad_left, pad_left + 16]:
            x = torch.randn(2, 4, N, requires_grad=True, device=device)
            x_pad = backend.pad(x, pad_left, pad_right)
            x_pad = x_pad.reshape(x_pad.shape[:-1])
            # Check the size
            x2 = x.clone()
            x_pad2 = x_pad.clone()
            # compare left reflected part of padded array with left side 
            # of original array
            for t in range(1, pad_left + 1):
                assert torch.allclose(x_pad2[..., pad_left - t], x2[..., t])
            # compare left part of padded array with left side of 
            # original array
            for t in range(x.shape[-1]):
                assert torch.allclose(x_pad2[..., pad_left + t], x2[..., t])
            # compare right reflected part of padded array with right side
            # of original array
            for t in range(1, pad_right + 1):
                assert torch.allclose(x_pad2[..., x_pad.shape[-1] - 1 - pad_right + t], x2[..., x.shape[-1] - 1 - t])
            # compare right part of padded array with right side of 
            # original array
            for t in range(1, pad_right + 1):
                assert torch.allclose(x_pad2[..., x_pad.shape[-1] - 1 - pad_right - t], x2[..., x.shape[-1] - 1 - t])

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
            assert torch.allclose(x.grad, x_grad)

    # Check that the padding shows an error if we try to pad
    with pytest.raises(ValueError) as ve:
        backend.pad(x, x.shape[-1], 0)
    assert "padding size" in ve.value.args[0]
    
    with pytest.raises(ValueError) as ve:
        backend.pad(x, 0, x.shape[-1])
    assert "padding size" in ve.value.args[0]


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_modulus(device, backend, random_state=42):
    """
    Tests the stability and differentiability of modulus
    """
    if backend.name.endswith('_skcuda') and device == "cpu":
        with pytest.raises(TypeError) as re:
            x_bad = torch.randn((4, 2)).cpu()
            backend.modulus(x_bad)
        assert "for CPU tensors" in re.value.args[0]
        return

    torch.manual_seed(random_state)
    # Test with a random vector
    x = torch.randn(2, 4, 128, 2, requires_grad=True, device=device)

    x_abs = backend.modulus(x).squeeze(-1)
    assert len(x_abs.shape) == len(x.shape[:-1])

    # check the value
    x_abs2 = x_abs.clone()
    x2 = x.clone()
    assert torch.allclose(x_abs2, torch.sqrt(x2[..., 0] ** 2 + x2[..., 1] ** 2))

    with pytest.raises(TypeError) as te:
        x_bad = torch.randn(4).to(device)
        backend.modulus(x_bad)
    assert "should be complex" in te.value.args[0]

    if backend.name.endswith('_skcuda'):
        pytest.skip("The skcuda backend does not pass differentiability"
            "tests, but that's ok (for now).")

    # check the gradient
    loss = torch.sum(x_abs)
    loss.backward()
    x_grad = x2 / x_abs2[..., None]
    assert torch.allclose(x.grad, x_grad)


    # Test the differentiation with a vector made of zeros
    x0 = torch.zeros(100, 4, 128, 2, requires_grad=True, device=device)
    x_abs0 = backend.modulus(x0)
    loss0 = torch.sum(x_abs0)
    loss0.backward()
    assert torch.max(torch.abs(x0.grad)) <= 1e-7


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("device", devices)
def test_subsample_fourier(backend, device, random_state=42):
    """
    Tests whether the periodization in Fourier performs a good subsampling
    in time
    """
    if backend.name.endswith('_skcuda') and device == 'cpu':
        with pytest.raises(TypeError) as re:
            x_bad = torch.randn((4, 2)).cpu()
            backend.subsample_fourier(x_bad, 1)
        assert "for CPU tensors" in re.value.args[0]
        return
    rng = np.random.RandomState(random_state)
    J = 10
    x = rng.randn(2, 4, 2**J) + 1j * rng.randn(2, 4, 2**J)
    x_f = np.fft.fft(x, axis=-1)[..., np.newaxis]
    x_f.dtype = 'float64'  # make it a vector
    x_f_th = torch.from_numpy(x_f).to(device)

    for j in range(J + 1):
        x_f_sub_th = backend.subsample_fourier(x_f_th, 2**j).cpu()
        x_f_sub = x_f_sub_th.numpy()
        x_f_sub.dtype = 'complex128'
        x_sub = np.fft.ifft(x_f_sub[..., 0], axis=-1)
        assert np.allclose(x[:, :, ::2**j], x_sub)

    # If we are using a GPU-only backend, make sure it raises the proper
    # errors for CPU tensors.
    if device=='cuda':
        with pytest.raises(TypeError) as te:
            x_bad = torch.randn(4).cuda()
            backend.subsample_fourier(x_bad, 1)
        assert "should be complex" in te.value.args[0]


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("device", devices)
def test_unpad(backend, device):
    if backend.name == "torch_skcuda" and device == "cpu":
        pytest.skip()

    # test unpading of a random tensor
    x = torch.randn(8, 4, 1).to(device)

    y = backend.unpad(x, 1, 3)

    assert y.shape == (8, 2)
    assert torch.allclose(y, x[:, 1:3, 0])

    N = 128
    x = torch.rand(2, 4, N).to(device)

    # similar to for loop in pad test
    for pad_left in range(0, N - 16, 16):
        pad_right = pad_left + 16
        x_pad = backend.pad(x, pad_left, pad_right)
        x_unpadded = backend.unpad(x_pad, pad_left, x_pad.shape[-1] - pad_right - 1)
        assert torch.allclose(x, x_unpadded)


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("device", devices)
def test_fft_type(backend, device):
    if backend.name == "torch_skcuda" and device == "cpu":
        pytest.skip()

    x = torch.randn(8, 4, 2).to(device)

    with pytest.raises(TypeError) as record:
        y = backend.rfft(x)
    assert 'should be real' in record.value.args[0]

    x = torch.randn(8, 4, 1).to(device)

    with pytest.raises(TypeError) as record:
        y = backend.ifft(x)
    assert 'should be complex' in record.value.args[0]

    with pytest.raises(TypeError) as record:
        y = backend.irfft(x)
    assert 'should be complex' in record.value.args[0]


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("device", devices)
def test_fft(backend, device):
    if backend.name == "torch_skcuda" and device == "cpu":
        pytest.skip()

    def coefficient(n):
            return np.exp(-2 * np.pi * 1j * n)

    x_r = np.random.rand(4)

    I, K = np.meshgrid(np.arange(4), np.arange(4), indexing='ij')

    coefficients = coefficient(K * I / x_r.shape[0])

    y_r = (x_r * coefficients).sum(-1)

    x_r = torch.from_numpy(x_r)[..., None].to(device)
    y_r = torch.from_numpy(np.column_stack((y_r.real, y_r.imag))).to(device)
    
    z = backend.rfft(x_r)
    assert torch.allclose(y_r, z)

    z_1 = backend.ifft(z)
    assert torch.allclose(x_r[..., 0], z_1[..., 0])

    z_2 = backend.irfft(z)
    assert not z_2.shape[-1] == 2
    assert torch.allclose(x_r, z_2)
    
    z_3 = backend.cfft(backend.ifft(z))
    assert torch.allclose(z, z_3)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_pad_frequency(device, backend, random_state=42):
    """
    Tests the correctness of pad_frequency
    """
    # Real (last dimension is 1)
    shape = (10, 20, 3, 5, 1)
    shape_padded = (10, 20, 3, 16, 1)
    x = torch.arange(np.prod(shape)).reshape(shape) * 0.5
    x_padded = backend.pad_frequency(x, padding=11)
    assert tuple(x_padded.shape) == shape_padded

    # Complex (trailing dimension is 2)
    shape = (10, 20, 3, 5, 2)
    shape_padded = (10, 20, 3, 16, 2)
    x = torch.arange(np.prod(shape)).reshape(shape) * 0.5
    x_padded = backend.pad_frequency(x, padding=11)
    assert tuple(x_padded.shape) == shape_padded


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_swap_time_frequency_1d(device, backend, random_state=42):
    """
    Tests the correctness of swap_time_frequency
    """
    shape = (10, 20, 3, 5, 1)
    shape_T = (10, 20, 5, 3, 1)

    x = torch.arange(np.prod(shape)).reshape(shape) * 0.5
    x_T = backend.swap_time_frequency(x)
    assert tuple(x_T.shape) == shape_T
    assert x_T.is_contiguous()

    x_T_T = backend.swap_time_frequency(x_T)
    assert tuple(x_T_T.shape) == shape
    assert x_T_T.shape == x.shape
    assert torch.all(x == x_T_T)
    assert x_T_T.is_contiguous()

    shape = (10, 20, 3, 5, 2)
    shape_T = (10, 20, 5, 3, 2)

    x = torch.arange(np.prod(shape)).reshape(shape)
    x_T = backend.swap_time_frequency(x)
    assert tuple(x_T.shape) == shape_T

    x_T_T = backend.swap_time_frequency(x_T)
    assert tuple(x_T_T.shape) == shape
    assert x_T_T.shape == x.shape
    assert torch.all(x == x_T_T)


def test_unpad_frequency():
    shape = (10, 20, 16, 3)
    shape_unpadded = (10, 20, 6, 3)
    x = np.arange(np.prod(shape)).reshape(shape) * 0.5
    x_unpadded = backend.unpad_frequency(x, n1_max=10, n1_stride=2)
    assert x_unpadded.shape == shape_unpadded


def test_split_frequency_axis():
    shape = (10, 20, 16, 3)
    x = torch.arange(np.prod(shape)).reshape(shape) * 0.5
    X_split = backend.split_frequency_axis(x)
    assert len(X_split) == x.shape[-3]
    for i, x_split in enumerate(X_split):
        assert x_split.shape[:-3] == x.shape[:-3]
        assert x_split.shape[-2:] == x.shape[-2:]
        assert torch.allclose(x_split[..., 0, :, :], x[..., i, :, :])
