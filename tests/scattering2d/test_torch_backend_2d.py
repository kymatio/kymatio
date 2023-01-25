""" This script will test the submodules used by the scattering module"""
import numpy as np
import torch
import pytest
from collections import namedtuple


devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')


backends = []
backends_devices = []

skcuda_available = False
try:
    if torch.cuda.is_available():
        from skcuda import cublas
        import cupy
        skcuda_available = True
except:
    Warning('torch_skcuda backend not available.')

if skcuda_available:
    from kymatio.scattering2d.backend.torch_skcuda_backend import backend
    backends.append(backend)
    if 'cuda' in devices:
        backends_devices.append((backend, 'cuda'))


from kymatio.scattering2d.backend.torch_backend import backend
backends.append(backend)
backends_devices.append((backend, 'cpu'))
if 'cuda' in devices:
    backends_devices.append((backend, 'cuda'))


class TestPad:
    @pytest.mark.parametrize('backend_device', backends_devices)
    def test_Pad(self, backend_device):
        backend, device = backend_device

        pad = backend.Pad((2, 2, 2, 2), (4, 4))

        x = torch.randn(1, 4, 4)
        x = x.to(device)

        z = pad(x)

        assert z.shape == (1, 8, 8, 1)
        assert torch.allclose(z[0, 2, 2], x[0, 0, 0])
        assert torch.allclose(z[0, 1, 0], x[0, 1, 2])
        assert torch.allclose(z[0, 1, 1], x[0, 1, 1])
        assert torch.allclose(z[0, 1, 2], x[0, 1, 0])
        assert torch.allclose(z[0, 1, 3], x[0, 1, 1])

    @pytest.mark.parametrize('backend_device', backends_devices)
    def test_unpad(self, backend_device):
        backend, device = backend_device

        x = torch.randn(4, 4, 1)
        x = x.to(device)

        y = backend.unpad(x)

        assert y.shape == (2, 2)
        assert torch.allclose(y[0, 0], x[1, 1, 0])
        assert torch.allclose(y[0, 1], x[1, 2, 0])


# Checked the modulus
class TestModulus:
    @pytest.mark.parametrize('backend_device', backends_devices)
    def test_Modulus(self, backend_device):
        backend, device = backend_device

        modulus = backend.modulus
        x = torch.rand(100, 10, 4, 2).to(device)

        y = modulus(x)
        y = y.reshape(y.shape[:-1])
        u = torch.sqrt(torch.sum(x * x, 3))
        assert torch.allclose(u, y)

        y = x[..., 0].contiguous()
        with pytest.raises(TypeError) as record:
            modulus(y)
        assert 'should be complex' in record.value.args[0]

        y = x[::2, ::2]
        with pytest.raises(RuntimeError) as record:
            modulus(y)
        assert 'contiguous' in record.value.args[0]

    @pytest.mark.parametrize('backend', backends)
    def test_cuda_only(self, backend):
        modulus = backend.modulus
        if backend.name == 'torch_skcuda':
            x = torch.rand(100, 10, 4, 2).cpu()
            with pytest.raises(TypeError) as exc:
                y = modulus(x)
            assert 'Use the torch backend' in exc.value.args[0]
   

# Checked the subsampling
class TestSubsampleFourier:
    @pytest.mark.parametrize('backend_device', backends_devices)
    def test_SubsampleFourier(self, backend_device):
        backend, device = backend_device
        subsample_fourier = backend.subsample_fourier

        x = torch.rand(100, 1, 128, 128, 2).to(device)

        y = torch.zeros(100, 1, 8, 8, 2).to(device)

        for i in range(8):
            for j in range(8):
                for m in range(16):
                    for n in range(16):
                        y[...,i,j,:] += x[...,i+m*8,j+n*8,:]

        y = y / (16*16)

        z = subsample_fourier(x, k=16)
        assert torch.allclose(y, z)

        # Must clone to make sure result is contiguous.
        y = x[..., 0].clone()
        with pytest.raises(TypeError) as record:
            subsample_fourier(y, k=16)
        assert 'should be complex' in record.value.args[0]

        y = x[::2, ::2]
        with pytest.raises(RuntimeError) as record:
            subsample_fourier(y, k=16)
        assert 'must be contiguous' in record.value.args[0]

    @pytest.mark.parametrize('backend', backends)
    def test_gpu_only(self, backend):
        subsample_fourier = backend.subsample_fourier

        if backend.name == 'torch_skcuda':
            x = torch.rand(100, 1, 128, 128, 2).cpu()
            with pytest.raises(TypeError) as exc:
                z = subsample_fourier(x, k=16)
            assert 'Use the torch backend' in exc.value.args[0]

    @pytest.mark.parametrize('backend_device', backends_devices)
    def test_batch_shape_agnostic(self, backend_device):
        backend, device = backend_device
        subsample_fourier = backend.subsample_fourier

        x = torch.rand(100, 1, 8, 128, 128, 2).to(device)

        y = torch.zeros(100, 1, 8, 8, 8, 2).to(device)

        for i in range(8):
            for j in range(8):
                for m in range(16):
                    for n in range(16):
                        y[...,i,j,:] += x[...,i+m*8,j+n*8,:]

        y = y / (16*16)

        z = subsample_fourier(x, k=16)
        assert torch.allclose(y, z)


# Check the CUBLAS routines
class TestCDGMM:
    @pytest.fixture(params=(False, True))
    def data(self, request):
        real_filter = request.param

        x = torch.rand(100, 128, 128, 2).float()
        filt = torch.rand(128, 128, 2).float()
        y = torch.ones(100, 128, 128, 2).float()

        if real_filter:
            filt[..., 1] = 0

        y[..., 0] = x[..., 0] * filt[..., 0] - x[..., 1] * filt[..., 1]
        y[..., 1] = x[..., 1] * filt[..., 0] + x[..., 0] * filt[..., 1]

        if real_filter:
            filt = filt[..., :1].contiguous()

        return x, filt, y

    @pytest.mark.parametrize('backend_device', backends_devices)
    def test_cdgmm_forward(self, data, backend_device):
        backend, device = backend_device

        x, filt, y = data
        x, filt, y = x.to(device), filt.to(device), y.to(device)

        z = backend.cdgmm(x, filt)

        Warning('Tolerance has been slightly lowered here...')
        # There is a very small meaningless difference for skcuda+GPU
        assert torch.allclose(y, z, atol=1e-7, rtol =1e-6)

    @pytest.mark.parametrize('backend', backends)
    def test_gpu_only(self, data, backend):
        x, filt, y = data
        if backend.name == 'torch_skcuda':
            x = x.cpu()
            filt = filt.cpu()

            with pytest.raises(TypeError) as exc:
                z = backend.cdgmm(x, filt)
            assert 'must be CUDA' in exc.value.args[0]

    @pytest.mark.parametrize('backend', backends)
    def test_cdgmm_exceptions(self, backend):
        with pytest.raises(RuntimeError) as exc:
            backend.cdgmm(torch.empty(3, 4, 5, 2), torch.empty(4, 3, 2))
        assert 'not compatible' in exc.value.args[0]

        with pytest.raises(TypeError) as exc:
            backend.cdgmm(torch.empty(3, 4, 5, 1), torch.empty(4, 5, 1))
        assert 'should be complex' in exc.value.args[0]

        with pytest.raises(TypeError) as exc:
            backend.cdgmm(torch.empty(3, 4, 5, 2), torch.empty(4, 5, 3))
        assert 'should be complex' in exc.value.args[0]

        with pytest.raises(TypeError) as exc:
            backend.cdgmm(torch.empty(3, 4, 5, 2),
                          torch.empty(4, 5, 1).double())
        assert 'must be of the same dtype' in exc.value.args[0]

        if 'cuda' in devices:
            if backend.name=='torch_skcuda':
                with pytest.raises(TypeError) as exc:
                    backend.cdgmm(torch.empty(3, 4, 5, 2),
                                  torch.empty(4, 5, 1).cuda())
                assert 'must be cuda tensors' in exc.value.args[0].lower()
            elif backend.name=='torch':
                with pytest.raises(TypeError) as exc:
                    backend.cdgmm(torch.empty(3, 4, 5, 2),
                                  torch.empty(4, 5, 1).cuda())
                assert 'input must be on gpu' in exc.value.args[0].lower()

                with pytest.raises(TypeError) as exc:
                    backend.cdgmm(torch.empty(3, 4, 5, 2).cuda(),
                                  torch.empty(4, 5, 1))
                assert 'input must be on cpu' in exc.value.args[0].lower()

    @pytest.mark.parametrize('backend_device', backends_devices)
    def test_contiguity_exception(self, backend_device):
        backend, device = backend_device

        x = torch.empty(3, 4, 5, 3).to(device)[..., :2]
        y = torch.empty(4, 5, 3).to(device)[..., :2]

        with pytest.raises(RuntimeError) as exc:
            backend.cdgmm(x.contiguous(), y)
        assert 'be contiguous' in exc.value.args[0]

        with pytest.raises(RuntimeError) as exc:
            backend.cdgmm(x, y.contiguous())
        assert 'be contiguous' in exc.value.args[0]

    @pytest.mark.parametrize('backend_device', backends_devices)
    def test_device_mismatch(self, backend_device):
        backend, device = backend_device

        if device == 'cpu':
            return

        if torch.cuda.device_count() < 2:
            return

        x = torch.empty(3, 4, 5, 2).to('cuda:0')
        y = torch.empty(4, 5, 1).to('cuda:1')

        with pytest.raises(TypeError) as exc:
            backend.cdgmm(x, y)
        assert 'must be on the same GPU' in exc.value.args[0]


class TestFFT:
    @pytest.mark.parametrize('backend', backends)
    def test_fft(self, backend):
        import numpy as np
        
        def coefficent(n):
            return np.exp(-2 * np.pi * 1j * n)

        x_r = np.random.rand(4, 4)

        I, J, K, L = np.meshgrid(np.arange(4), np.arange(4), np.arange(4),
                np.arange(4), indexing='ij')

        coefficents = coefficent(K * I / x_r.shape[0] + L * J / x_r.shape[1])

        y_r = np.zeros(x_r.shape).astype('complex128')
        
        for k in range(4):
            for l in range(4):
                y_r[k, l] = (x_r * coefficents[..., k, l]).sum()


        y_r = torch.from_numpy(np.stack((y_r.real, y_r.imag), axis=-1))
        x_r = torch.from_numpy(x_r)[..., None]


        z = backend.rfft(x_r)
        assert torch.allclose(y_r, z)
        
        z_1 = backend.irfft(z)
        assert z_1.shape == x_r.shape
        assert torch.allclose(x_r, z_1)


        z_2 = backend.ifft(z)[..., :1]
        assert torch.allclose(x_r, z_2)
        
         
    @pytest.mark.parametrize('backend_device', backends_devices)
    def test_fft_exceptions(self, backend_device):
        backend, device = backend_device

        x = torch.randn(4, 4, 2)
        x = x.to(device)
        with pytest.raises(TypeError) as record:
            backend.rfft(x)
        assert 'real' in record.value.args[0]

        x = torch.randn(4, 4, 1)
        x = x.to(device)
        with pytest.raises(TypeError) as record:
            backend.ifft(x)
        assert 'complex' in record.value.args[0]
       
        x = torch.randn(4, 4, 1)
        x = x.to(device)
        with pytest.raises(TypeError) as record:
            backend.irfft(x)
        assert 'complex' in record.value.args[0]
       
        x = torch.randn(4, 4, 1)
        x = x.to(device)
        y = x[::2, ::2]

        with pytest.raises(RuntimeError) as record:
            backend.rfft(y)
        assert 'must be contiguous' in record.value.args[0]
        
        x = torch.randn(4, 4, 2)
        x = x.to(device)
        y = x[::2, ::2]

        with pytest.raises(RuntimeError) as record:
            backend.ifft(y)
        assert 'must be contiguous' in record.value.args[0]

        x = torch.randn(4, 4, 2)
        x = x.to(device)
        y = x[::2, ::2]

        with pytest.raises(RuntimeError) as record:
            backend.irfft(y)
        assert 'must be contiguous' in record.value.args[0]


class TestBackendUtils:
    @pytest.mark.parametrize('backend', backends)
    def test_stack(self, backend):
        x = torch.randn(3, 6, 6)
        y = torch.randn(3, 6, 6)
        z = torch.randn(3, 6, 6)

        w = backend.stack((x, y, z))

        assert w.shape == (x.shape[0],) + (3,) + (x.shape[-2:])
        assert np.allclose(w[:, 0, ...], x)
        assert np.allclose(w[:, 1, ...], y)
        assert np.allclose(w[:, 2, ...], z)
