""" This script will test the submodules used by the scattering module"""

import os
import numpy as np
import torch
import pytest
from kymatio.scattering2d import Scattering2D_torch as Scattering2D



backends = []

try:
    if torch.cuda.is_available():
        from skcuda import cublas
        import cupy
        from kymatio.scattering2d.backend import torch_skcuda_backend as skcuda_b
        backends.append(skcuda_b)
except:
    pass

try:
    from kymatio.scattering2d.backend import torch_backend as torch_backend
    backends.append(torch_backend)
except:
    pass


if torch.cuda.is_available():
    devices = ['cuda', 'cpu']
else:
    devices = ['cpu']


# Checked the modulus
class TestModulus:
    @pytest.mark.parametrize("device", devices)
    def test_Modulus(self, device):
        if device == 'cuda':
            for backend in backends:
                modulus = backend.modulus
                x = torch.rand(100, 10, 4, 2).cuda().float()
                y = modulus(x)
                u = torch.squeeze(torch.sqrt(torch.sum(x * x, 3)))
                v = y.narrow(3, 0, 1)
                u = u.squeeze()
                v = v.squeeze()
                assert torch.allclose(u, v)
        elif device == 'cpu':
            for backend in backends:
                if backend.name == 'skcuda':
                    continue
                modulus = backend.modulus
                x = torch.rand(100, 10, 4, 2).float()
                y = modulus(x)
                u = torch.squeeze(torch.sqrt(torch.sum(x * x, 3)))
                v = y.narrow(3, 0, 1)
                u = u.squeeze()
                v = v.squeeze()
                assert torch.allclose(u, v)


# Checked the subsampling
class TestSubsampleFourier:
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("backend", backends)
    def test_SubsampleFourier(self, device, backend):
        if device == 'cuda':
            for backend in backends:
                x = torch.rand(100, 1, 128, 128, 2).cuda().double()
                y = torch.zeros(100, 1, 8, 8, 2).cuda().double()

                for i in range(8):
                    for j in range(8):
                        for m in range(16):
                            for n in range(16):
                                y[...,i,j,:] += x[...,i+m*8,j+n*8,:]

                y = y / (16*16)

                subsample_fourier = backend.subsample_fourier

                z = subsample_fourier(x, k=16)
                assert torch.allclose(y, z)
                if backend.name == 'torch':
                    z = subsample_fourier(x.cpu(), k=16)
                    assert torch.allclose(y, z)
        elif device == 'cpu' and backend.name != 'skcuda':
            x = torch.rand(100, 1, 128, 128, 2).cpu().double()
            y = torch.zeros(100, 1, 8, 8, 2).cpu().double()

            for i in range(8):
                for j in range(8):
                    for m in range(16):
                        for n in range(16):
                            y[...,i,j,:] += x[...,i+m*8,j+n*8,:]

            y = y / (16*16)

            subsample_fourier = backend.subsample_fourier

            z = subsample_fourier(x, k=16)
            assert torch.allclose(y, z)
            if backend.name == 'torch':
                z = subsample_fourier(x.cpu(), k=16)
                assert torch.allclose(y, z)


# Check the CUBLAS routines
class TestCDGMM:
    @pytest.fixture(params=(False, True))
    def data(self, request):
        real_filter = request.param
        x = torch.rand(100, 128, 128, 2)
        filt = torch.rand(128, 128, 2)
        y = torch.ones(100, 128, 128, 2)
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
        if device == 'cpu' and backend.name == 'skcuda':
            pytest.skip("skcuda backend can only run on gpu")
        x, filt, y = data
        # move to device
        x, filt, y = x.to(device), filt.to(device), y.to(device)
        # call cdgmm
        if inplace:
            x = x.clone()
        z = backend.cdgmm(x, filt, inplace=inplace)
        if inplace:
            z = x
        # compare
        assert torch.allclose(y, z)

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
        with pytest.raises(RuntimeError) as exc:
            backend.cdgmm(torch.empty(3, 4, 5, 2), torch.empty(4, 5, 1).double())
        assert "must be of the same dtype" in exc.value.args[0]
        if 'cuda' in devices:
            with pytest.raises(RuntimeError) as exc:
                backend.cdgmm(torch.empty(3, 4, 5, 2), torch.empty(4, 5, 1).cuda())
            assert "type" in exc.value.args[0]

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


class TestScattering2D_Torch:
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

    # Check the scattering
    # FYI: access the two different tests in here by setting envs
    # KYMATIO_BACKEND=skcuda and KYMATIO_BACKEND=torch
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

        if backend.name == 'skcuda':
            print('skcuda backend tested!')
            # First, let's check the Jit
            scattering = Scattering2D(J, shape=(M, N), pre_pad=pre_pad, backend=backend)
            scattering.cuda()
            x = x.cuda()
            S = S.cuda()
            y = scattering(x)
            assert ((S - y)).abs().max() < 1e-6
        elif backend.name == 'torch':
            # Then, let's check when using pure pytorch code
            scattering = Scattering2D(J, shape=(M, N), pre_pad=pre_pad, backend=backend)
            Sg = []
            if device == 'cuda':
                print('torch-gpu backend tested!')
                x = x.cuda()
                scattering.cuda()
                S = S.cuda()
                Sg = scattering(x)
            else:
                print('torch-cpu backend tested!')
                x = x.cpu()
                S = S.cpu()
                scattering.cpu()
                Sg = scattering(x)
            assert torch.allclose(Sg, S)

    @pytest.mark.parametrize("backend", backends)
    def test_batch_shape_agnostic(self, backend):
        J = 3
        L = 8
        shape = (32, 32)

        shape_ds = tuple(n // 2 **J for n in shape)

        S = Scattering2D(J, shape, L, backend=backend)

        with pytest.raises(RuntimeError) as ve:
            S(torch.zeros(()))
        assert "at least two" in ve.value.args[0]

        with pytest.raises(RuntimeError) as ve:
            S(torch.zeros((32, )))
        assert "at least two" in ve.value.args[0]

        x = torch.zeros(shape)

        if backend.name == 'skcuda':
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

            if backend.name == 'skcuda':
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
        S = Scattering2D(3, (32, 32), backend=backend)

        if backend.name == 'skcuda':
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

        S = Scattering2D(3, (32, 32), pre_pad=True, backend=backend)

        with pytest.raises(RuntimeError) as record:
            S(x)
        assert('Padded tensor must be of spatial size' in record.value.args[0])

        x = torch.randn(8,8)
        S = Scattering2D(2, (8, 8), backend=backend)


        x = x.to(device)
        S = S.to(device)
        if not (device == 'cpu' and backend.name == 'skcuda'):
            y = S(x)
            assert(x.device == y.device)

    # Check that several input size works
    @pytest.mark.parametrize("backend", backends)
    def test_input_size_agnostic(self, backend):
        for N in [31, 32, 33]:
            for J in [1, 2, 4]:
                scattering = Scattering2D(J, shape=(N, N), backend=backend)
                x = torch.zeros(3, 3, N, N)

                if backend.name == 'skcuda':
                    x = x.cuda()
                    scattering.cuda()

                S = scattering(x)
                scattering = Scattering2D(J, shape=(N, N), pre_pad=True, backend=backend)
                x = torch.zeros(3,3,scattering.M_padded, scattering.N_padded)

                if backend.name == 'skcuda':
                    x = x.cuda()
                    scattering.cuda()

                S = scattering(x)

        N = 32
        J = 5
        scattering = Scattering2D(J, shape=(N, N), backend=backend)
        x = torch.zeros(3, 3, N, N)

        if backend.name == 'skcuda':
            x = x.cuda()
            scattering.cuda()

        S = scattering(x)
        assert(S.shape[-2:] == (1, 1))

        N = 32
        J = 5
        scattering = Scattering2D(J, shape=(N+5, N), backend=backend)
        x = torch.zeros(3, 3, N+5, N)

        if backend.name == 'skcuda':
            x = x.cuda()
            scattering.cuda()

        S = scattering(x)
        assert (S.shape[-2:] == (1, 1))
