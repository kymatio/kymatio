""" This script will test the submodules used by the scattering module"""

import os
import io
import numpy as np
import torch
import pytest
from kymatio import Scattering2D
from torch.autograd import gradcheck
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


class TestScatteringTorch2D:
    @pytest.mark.parametrize('backend_device', backends_devices)
    def test_Scattering2D(self, backend_device):
        backend, device = backend_device

        test_data_dir = os.path.dirname(__file__)
        with open(os.path.join(test_data_dir, 'test_data_2d.npz'), 'rb') as f:
            buffer = io.BytesIO(f.read())
            data = np.load(buffer)

        x = torch.from_numpy(data['x'])
        S = torch.from_numpy(data['Sx'])
        J = data['J']
        pre_pad = data['pre_pad']

        M = x.shape[2]
        N = x.shape[3]

        scattering = Scattering2D(J, shape=(M, N), pre_pad=pre_pad,
                                  backend=backend, frontend='torch')
        Sg = []
        x = x.to(device)
        scattering.to(device)
        S = S.to(device)
        Sg = scattering(x)
        assert torch.allclose(Sg, S)

        scattering = Scattering2D(J, shape=(M, N), pre_pad=pre_pad,
                                  max_order=1, frontend='torch',
                                  backend=backend)
        scattering.to(device)

        S1x = scattering(x)
        assert torch.allclose(S1x, S[..., :S1x.shape[-3], :, :])

    @pytest.mark.parametrize('backend', backends)
    def test_gpu_only(self, backend):
        if backend.name == 'torch_skcuda':
            scattering = Scattering2D(3, shape=(32, 32), backend=backend,
                                      frontend='torch')

            x = torch.rand(32, 32)

            with pytest.raises(TypeError) as ve:
                Sg = scattering(x)
            assert 'CUDA' in ve.value.args[0]

    @pytest.mark.parametrize('backend_device', backends_devices)
    def test_batch_shape_agnostic(self, backend_device):
        backend, device = backend_device

        J = 3
        L = 8
        shape = (32, 32)

        shape_ds = tuple(n // (2 ** J) for n in shape)

        S = Scattering2D(J, shape, L, backend=backend, frontend='torch')

        with pytest.raises(RuntimeError) as ve:
            S(torch.zeros(()))
        assert 'at least two' in ve.value.args[0]

        with pytest.raises(RuntimeError) as ve:
            S(torch.zeros((32, )))
        assert 'at least two' in ve.value.args[0]

        x = torch.zeros(shape)

        x = x.to(device)
        S.to(device)

        Sx = S(x)

        assert len(Sx.shape) == 3
        assert Sx.shape[-2:] == shape_ds

        n_coeffs = Sx.shape[-3]

        test_shapes = ((1,) + shape, (2,) + shape, (2, 2) + shape,
                       (2, 2, 2) + shape)

        for test_shape in test_shapes:
            x = torch.zeros(test_shape)

            x = x.to(device)

            Sx = S(x)

            assert len(Sx.shape) == len(test_shape) + 1
            assert Sx.shape[-2:] == shape_ds
            assert Sx.shape[-3] == n_coeffs
            assert Sx.shape[:-3] == test_shape[:-2]

    @pytest.mark.parametrize('backend_device', backends_devices)
    def test_scattering2d_errors(self, backend_device):
        backend, device = backend_device

        S = Scattering2D(3, (32, 32), backend=backend, frontend='torch')

        S.to(device)

        with pytest.raises(TypeError) as record:
            S(None)
        assert 'input should be' in record.value.args[0]

        x = torch.randn(4,4)
        y = x[::2,::2]

        with pytest.raises(RuntimeError) as record:
            S(y)
        assert 'must be contiguous' in record.value.args[0]

        x = torch.randn(31, 31)

        with pytest.raises(RuntimeError) as record:
            S(x)
        assert 'Tensor must be of spatial size' in record.value.args[0]

        S = Scattering2D(3, (32, 32), pre_pad=True, backend=backend,
                         frontend='torch')

        with pytest.raises(RuntimeError) as record:
            S(x)
        assert 'Padded tensor must be of spatial size' in record.value.args[0]

        x = torch.randn(8,8)
        S = Scattering2D(2, (8, 8), backend=backend, frontend='torch')

        x = x.to(device)
        S = S.to(device)
        if not (device == 'cpu' and backend.name == 'torch_skcuda'):
            y = S(x)
            assert x.device == y.device

    @pytest.mark.parametrize('backend_device', backends_devices)
    def test_input_size_agnostic(self, backend_device):
        backend, device = backend_device

        for N in [31, 32, 33]:
            for J in [1, 2, 4]:
                scattering = Scattering2D(J, shape=(N, N), backend=backend,
                                          frontend='torch')
                x = torch.zeros(3, 3, N, N)

                x = x.to(device)
                scattering.to(device)

                S = scattering(x)
                scattering = Scattering2D(J, shape=(N, N), pre_pad=True,
                                          backend=backend, frontend='torch')
                x = torch.zeros(3, 3, scattering._M_padded, scattering._N_padded)

                x = x.to(device)
                scattering.to(device)

        N = 32
        J = 5
        scattering = Scattering2D(J, shape=(N, N), backend=backend,
                                  frontend='torch')
        x = torch.zeros(3, 3, N, N)

        x = x.to(device)
        scattering.to(device)

        S = scattering(x)
        assert S.shape[-2:] == (1, 1)

        N = 32
        J = 5
        scattering = Scattering2D(J, shape=(N+5, N), backend=backend,
                                  frontend='torch')
        x = torch.zeros(3, 3, N+5, N)

        x = x.to(device)
        scattering.to(device)

        S = scattering(x)
        assert S.shape[-2:] == (1, 1)

    def test_inputs(self):
        fake_backend = namedtuple('backend', ['name',])
        fake_backend.name = 'fake'

        with pytest.raises(ImportError) as ve:
            scattering = Scattering2D(2, shape=(10, 10), frontend='torch',
                                      backend=fake_backend)
        assert 'not supported' in ve.value.args[0]

        with pytest.raises(RuntimeError) as ve:
            scattering = Scattering2D(10, shape=(10, 10), frontend='torch')
        assert 'smallest dimension' in ve.value.args[0]

    @pytest.mark.parametrize('backend_device', backends_devices)
    def test_gradients(self, backend_device):
        backend, device = backend_device

        if backend.name == 'torch_skcuda':
            pytest.skip('The gradients are currently not implemented with '
                        'the skcuda backend.')
        else:
            scattering = Scattering2D(2, shape=(8, 8), backend=backend,
                                      frontend='torch').double().to(device)
            x = torch.rand(2, 1, 8, 8).double().to(device).requires_grad_()
            gradcheck(scattering, x, nondet_tol=1e-5, fast_mode=True)
