""" This script will test the submodules used by the scattering module"""

import os
import numpy as np
import torch
import pytest
from kymatio.scattering2d import Scattering2D
from torch.autograd import gradcheck
from collections import namedtuple
backends = []

try:
    if torch.cuda.is_available():
        from skcuda import cublas
        import cupy
        from kymatio.scattering2d.backend.torch_skcuda_backend import backend
        backends.append(backend)
except:
    pass


from kymatio.scattering2d.backend.torch_backend import backend
backends.append(backend)

if torch.cuda.is_available():
    devices = ['cuda', 'cpu']
else:
    devices = ['cpu']


# Checked the modulus
class TestModulus:
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("backend", backends)
    def test_Modulus(self, device, backend):
        modulus = backend.modulus
        x = torch.rand(100, 10, 4, 2).to(device)

        if device == 'cpu' and backend.name == 'torch_skcuda':
            with pytest.raises(TypeError) as exc:
                y = modulus(x)
            assert "Use the torch backend" in exc.value.args[0]
        else:
            y = modulus(x)
            u = torch.squeeze(torch.sqrt(torch.sum(x * x, 3)))
            v = y.narrow(3, 0, 1)
            u = u.squeeze()
            v = v.squeeze()
            assert torch.allclose(u, v)

            y = x[::2, ::2]
            if backend.name == 'torch_skcuda':
                with pytest.raises(RuntimeError) as record:
                    modulus(y)
                assert ('Input' in record.value.args[0])



# Checked the subsampling
class TestSubsampleFourier:
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("backend", backends)
    def test_SubsampleFourier(self, device, backend):
        x = torch.rand(100, 1, 128, 128, 2).to(device)
        subsample_fourier = backend.subsample_fourier
        if device == 'cpu' and backend.name == 'torch_skcuda':
            with pytest.raises(TypeError) as exc:
                z = subsample_fourier(x, k=16)
            assert "Use the torch backend" in exc.value.args[0]

        else:
            y = torch.zeros(100, 1, 8, 8, 2).to(device)

            for i in range(8):
                for j in range(8):
                    for m in range(16):
                        for n in range(16):
                            y[...,i,j,:] += x[...,i+m*8,j+n*8,:]

            y = y / (16*16)

            z = subsample_fourier(x, k=16)
            assert torch.allclose(y, z)
            if backend.name == 'torch':
                z = subsample_fourier(x, k=16)
                assert torch.allclose(y, z)


            y = x[::2, ::2]
            with pytest.raises(RuntimeError) as record:
                subsample_fourier(y, k=16)
            assert ('Input' in record.value.args[0])

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("backend", backends)
    def test_batch_shape_agnostic(self, device, backend):
        x = torch.rand(100, 1, 8, 128, 128, 2).to(device)
        subsample_fourier = backend.subsample_fourier
        if device == 'cpu' and backend.name == 'torch_skcuda':
            with pytest.raises(TypeError) as exc:
                z = subsample_fourier(x, k=16)
            assert "Use the torch backend" in exc.value.args[0]

        else:
            y = torch.zeros(100, 1, 8, 8, 8, 2).to(device)

            for i in range(8):
                for j in range(8):
                    for m in range(16):
                        for n in range(16):
                            y[...,i,j,:] += x[...,i+m*8,j+n*8,:]

            y = y / (16*16)

            z = subsample_fourier(x, k=16)
            print(z.size(), y.size())
            assert torch.allclose(y, z)
            if backend.name == 'torch':
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
            filt = filt[..., :1]
        return x, filt, y

    @pytest.mark.parametrize("backend", backends)
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("inplace", (False, True))
    def test_cdgmm_forward(self, data, backend, device, inplace):
        x, filt, y = data
        # move to device
        x, filt, y = x.to(device), filt.to(device), y.to(device)
        if device == 'cpu' and backend.name == 'torch_skcuda':
            with pytest.raises(TypeError) as exc:
                z = backend.cdgmm(x, filt, inplace=inplace)
            assert "must be cuda" in exc.value.args[0]
        else:
            z = backend.cdgmm(x, filt, inplace=inplace)
            # compare
            Warning('Tolerance has been slightly lowered here...')
            assert torch.allclose(y, z, atol=1e-7, rtol =1e-6) # There is a very small meaningless difference for skcuda+GPU

    @pytest.mark.parametrize("backend", backends)
    def test_cdgmm_exceptions(self, backend):
        with pytest.raises(RuntimeError) as exc:
            backend.cdgmm(torch.empty(3, 4, 5, 2), torch.empty(4, 3, 2))
        assert "not compatible" in exc.value.args[0]
        with pytest.raises(TypeError) as exc:
            backend.cdgmm(torch.empty(3, 4, 5, 1), torch.empty(4, 5, 1))
        assert "input must be complex" in exc.value.args[0]
        with pytest.raises(TypeError) as exc:
            backend.cdgmm(torch.empty(3, 4, 5, 2), torch.empty(4, 5, 3))
        assert "filter must be complex or real" in exc.value.args[0]
        with pytest.raises(RuntimeError) as exc:
            backend.cdgmm(torch.empty(3, 4, 5, 2), torch.empty(3, 4, 5, 2))
        assert "filter must be a 3-tensor" in exc.value.args[0]
        with pytest.raises(TypeError) as exc:
            backend.cdgmm(torch.empty(3, 4, 5, 2), torch.empty(4, 5, 1).double())
        assert "must be of the same dtype" in exc.value.args[0]
        if 'cuda' in devices:
            if backend.name=='torch_skcuda':
                with pytest.raises(TypeError) as exc:
                    backend.cdgmm(torch.empty(3, 4, 5, 2), torch.empty(4, 5, 1).cuda())
                assert "cuda" in exc.value.args[0].lower()
            elif backend.name=='torch':
                with pytest.raises(TypeError) as exc:
                    backend.cdgmm(torch.empty(3, 4, 5, 2), torch.empty(4, 5, 1).cuda())
                assert "a must" in exc.value.args[0].lower()

class TestFFT:
    @pytest.mark.parametrize("backend", backends)
    def test_fft(self, backend):
        x = torch.rand(4, 4, 1)
        with pytest.raises(TypeError) as record:
            backend.fft(x)
        assert ('complex' in record.value.args[0])

        x = torch.randn(4, 4, 2)
        y = x[::2, ::2]

        with pytest.raises(RuntimeError) as record:
            backend.fft(y)
        assert ('must be contiguous' in record.value.args[0])


class TestScattering2DTorch:
    def reorder_coefficients_from_interleaved(self, J, L):
        # helper function to obtain positions of order0, order1, order2 from interleaved
        order0, order1, order2 = [], [], []
        n_order0, n_order1, n_order2 = 1, J * L, L ** 2 * J * (J - 1) // 2
        n = 0
        order0.append(n)
        for j1 in range(J):
            for l1 in range(L):
                n += 1
                order1.append(n)
                for j2 in range(j1 + 1, J):
                    for l2 in range(L):
                        n += 1
                        order2.append(n)
        assert len(order0) == n_order0
        assert len(order1) == n_order1
        assert len(order2) == n_order2
        return order0, order1, order2

    @pytest.mark.parametrize("backend", backends)
    @pytest.mark.parametrize("device", devices)
    def test_Scattering2D(self, backend, device):
        test_data_dir = os.path.dirname(__file__)
        data = torch.load(os.path.join(test_data_dir, 'test_data_2d.pt'))

        x = data['x']
        S = data['Sx']
        J = data['J']

        # we need to reorder S from interleaved (how it's saved) to o0, o1, o2
        # (which is how it's now computed)

        o0, o1, o2 = self.reorder_coefficients_from_interleaved(J, L=8)
        reorder = torch.from_numpy(np.concatenate((o0, o1, o2)))
        S = S[..., reorder, :, :]

        pre_pad = data['pre_pad']

        M = x.shape[2]
        N = x.shape[3]

        # Then, let's check when using pure pytorch code
        scattering = Scattering2D(J, shape=(M, N), pre_pad=pre_pad, backend=backend, frontend='torch')
        Sg = []
        x = x.to(device)
        scattering.to(device)
        S = S.to(device)
        if backend.name == 'torch_skcuda' and device == 'cpu':
            with pytest.raises(TypeError) as ve:
                Sg = scattering(x)
            assert "cuda" in ve.value.args[0]
        else:
            Sg = scattering(x)
            assert torch.allclose(Sg, S)

        # check also the default backend
        scattering = Scattering2D(J, shape=(M, N), pre_pad=pre_pad, frontend='torch')
        Sg = []
        x = x.to(device)
        scattering.to(device)
        S = S.to(device)
        Sg = scattering(x)
        assert torch.allclose(Sg, S)

    @pytest.mark.parametrize("backend", backends)
    def test_batch_shape_agnostic(self, backend):
        J = 3
        L = 8
        shape = (32, 32)

        shape_ds = tuple(n // 2 **J for n in shape)

        S = Scattering2D(J, shape, L, backend=backend, frontend='torch')

        with pytest.raises(RuntimeError) as ve:
            S(torch.zeros(()))
        assert "at least two" in ve.value.args[0]

        with pytest.raises(RuntimeError) as ve:
            S(torch.zeros((32, )))
        assert "at least two" in ve.value.args[0]

        x = torch.zeros(shape)

        if backend.name == 'torch_skcuda':
            x = x.cuda()
            S.cuda()

        Sx = S(x)

        assert len(Sx.shape) == 3
        assert Sx.shape[-2:] == shape_ds

        n_coeffs = Sx.shape[-3]

        test_shapes = ((1,) + shape, (2,) + shape, (2, 2) + shape,
                       (2, 2, 2) + shape)

        for test_shape in test_shapes:
            x = torch.zeros(test_shape)

            if backend.name == 'torch_skcuda':
                x = x.cuda()

            Sx = S(x)

            assert len(Sx.shape) == len(test_shape) + 1
            assert Sx.shape[-2:] == shape_ds
            assert Sx.shape[-3] == n_coeffs
            assert Sx.shape[:-3] == test_shape[:-2]

    # Make sure we test for the errors that may be raised by
    # `Scattering2D.forward`.
    @pytest.mark.parametrize("backend", backends)
    @pytest.mark.parametrize("device", devices)
    def test_scattering2d_errors(self, backend, device):
        S = Scattering2D(3, (32, 32), backend=backend, frontend='torch')

        if backend.name == 'torch_skcuda':
            S.cuda()

        with pytest.raises(TypeError) as record:
            S(None)
        assert('input should be' in record.value.args[0])

        x = torch.randn(4,4)
        y = x[::2,::2]

        with pytest.raises(RuntimeError) as record:
            S(y)
        assert('must be contiguous' in record.value.args[0])

        x = torch.randn(31, 31)

        with pytest.raises(RuntimeError) as record:
            S(x)
        assert('Tensor must be of spatial size' in record.value.args[0])

        S = Scattering2D(3, (32, 32), pre_pad=True, backend=backend, frontend='torch')

        with pytest.raises(RuntimeError) as record:
            S(x)
        assert('Padded tensor must be of spatial size' in record.value.args[0])

        x = torch.randn(8,8)
        S = Scattering2D(2, (8, 8), backend=backend, frontend='torch')


        x = x.to(device)
        S = S.to(device)
        if not (device == 'cpu' and backend.name == 'torch_skcuda'):
            y = S(x)
            assert(x.device == y.device)

    # Check that several input size works
    @pytest.mark.parametrize("backend", backends)
    def test_input_size_agnostic(self, backend):
        for N in [31, 32, 33]:
            for J in [1, 2, 4]:
                scattering = Scattering2D(J, shape=(N, N), backend=backend, frontend='torch')
                x = torch.zeros(3, 3, N, N)

                if backend.name == 'torch_skcuda':
                    x = x.cuda()
                    scattering.cuda()

                S = scattering(x)
                scattering = Scattering2D(J, shape=(N, N), pre_pad=True, backend=backend, frontend='torch')
                x = torch.zeros(3,3,scattering.M_padded, scattering.N_padded)

                if backend.name == 'torch_skcuda':
                    x = x.cuda()
                    scattering.cuda()

                S = scattering(x)

        N = 32
        J = 5
        scattering = Scattering2D(J, shape=(N, N), backend=backend, frontend='torch')
        x = torch.zeros(3, 3, N, N)

        if backend.name == 'torch_skcuda':
            x = x.cuda()
            scattering.cuda()

        S = scattering(x)
        assert(S.shape[-2:] == (1, 1))

        N = 32
        J = 5
        scattering = Scattering2D(J, shape=(N+5, N), backend=backend, frontend='torch')
        x = torch.zeros(3, 3, N+5, N)

        if backend.name == 'torch_skcuda':
            x = x.cuda()
            scattering.cuda()

        S = scattering(x)
        assert (S.shape[-2:] == (1, 1))

    def test_inputs(self):
        fake_backend = namedtuple('backend', ['name',])
        fake_backend.name = 'fake'

        with pytest.raises(ImportError) as ve:
            scattering = Scattering2D(2, shape=(10, 10), frontend='torch', backend=fake_backend)
        assert 'not supported' in ve.value.args[0]

        with pytest.raises(RuntimeError) as ve:
            scattering = Scattering2D(10, shape=(10, 10), frontend='torch')
        assert 'smallest dimension' in ve.value.args[0]

    @pytest.mark.parametrize("backend", backends)
    @pytest.mark.parametrize("device", devices)
    def test_gradients(self, backend, device):
        if backend.name == 'torch_skcuda':
            pytest.skip("The gradients are currently not implemented with the skcuda backend.")
        else:
            scattering = Scattering2D(2, shape=(8, 8), backend=backend, frontend='torch').double().to(device)
            x = torch.rand(2, 1, 8, 8).double().to(device).requires_grad_()
            gradcheck(scattering, x, nondet_tol=1e-5)
