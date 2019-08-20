import pytest
import torch
from kymatio import Scattering1D
import math
import os
import numpy as np


backends = []

try:
    if torch.cuda.is_available():
        from skcuda import cublas
        import cupy
        from kymatio.scattering1d.backend import torch_skcuda_backend
        backends.append(torch_skcuda_backend)
except:
    pass

try:
    from kymatio.scattering1d.backend import torch_backend
    backends.append(torch_backend)
except:
    pass


if torch.cuda.is_available():
    devices = ['cuda', 'cpu']
else:
    devices = ['cpu']

@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_simple_scatterings(device, backend, random_state=42):
    """
    Checks the behaviour of the scattering on simple signals
    (zero, constant, pure cosine)
    """
    rng = np.random.RandomState(random_state)
    J = 6
    Q = 8
    T = 2**12
    scattering = Scattering1D(J, T, Q, backend=backend, frontend='torch').to(device)

    # zero signal
    x0 = torch.zeros(128, T).to(device)

    s = scattering(x0)

    # check that s is zero!
    assert torch.max(torch.abs(s)) < 1e-7

    # constant signal
    x1 = rng.randn(1)[0] * torch.ones(1, T).to(device)
    s1 = scattering(x1)

    # check that all orders above 1 are 0
    assert torch.max(torch.abs(s1[:, 1:])) < 1e-7

    # sinusoid scattering
    meta = scattering.meta()
    for _ in range(50):
        k = rng.randint(1, T // 2, 1)[0]
        x2 = torch.cos(2 * math.pi * float(k) * torch.arange(0, T, dtype=torch.float32) / float(T))
        x2 = x2.unsqueeze(0)
        x2.to(device)
        s2 = scattering(x2)

        assert(s2[:,torch.from_numpy(meta['order']) != 1,:].abs().max() < 1e-2)

@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_sample_scattering(device, backend):
    """
    Applies scattering on a stored signal to make sure its output agrees with
    a previously calculated version.
    """
    test_data_dir = os.path.dirname(__file__)
    test_data_filename = os.path.join(test_data_dir, 'test_data_1d.pt')
    data = torch.load(test_data_filename, map_location='cpu')

    x = data['x']
    J = data['J']
    Q = data['Q']
    Sx0 = data['Sx']

    T = x.shape[2]

    # Convert from old (B, 1, T) format.
    x = x.squeeze(1).to(device)

    scattering = Scattering1D(J, T, Q, backend=backend, frontend='torch').to(device)

    # Reorder reference scattering from interleaved to concatenated orders.
    meta = scattering.meta()

    orders = [[], [], []]

    ind = 0

    orders[0].append(ind)
    ind = ind + 1

    n1s = [meta['key'][k][0] for k in range(len(meta['key']))
           if meta['order'][k] == 1]
    for n1 in n1s:
        orders[1].append(ind)
        ind = ind + 1

        n2s = [meta['key'][k][1] for k in range(len(meta['key']))
               if meta['order'][k] == 2 and meta['key'][k][0] == n1]

        for n2 in n2s:
            orders[2].append(ind)
            ind = ind + 1

    perm = torch.from_numpy(np.concatenate(orders))

    Sx0 = Sx0[:,perm,:]

    Sx = scattering(x)

    assert (Sx - Sx0).abs().max() < 1e-6


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_computation_Ux(backend, device, random_state=42):
    """
    Checks the computation of the U transform (no averaging for 1st order)
    """
    rng = np.random.RandomState(random_state)
    J = 6
    Q = 8
    T = 2**12
    scattering = Scattering1D(J, T, Q, average=False,
                              max_order=1, vectorize=False, frontend='torch', backend=backend).to(device)
    # random signal
    x = torch.from_numpy(rng.randn(1, T)).float().to(device)

    s = scattering(x)

    # check that the keys in s correspond to the order 0 and second order
    for k in range(len(scattering.psi1_f)):
        assert (k,) in s.keys()
    for k in s.keys():
        if k is not ():
            assert k[0] < len(scattering.psi1_f)
        else:
            assert True

    scattering.max_order = 2

    s = scattering(x)

    count = 1
    for k1, filt1 in enumerate(scattering.psi1_f):
        assert (k1,) in s.keys()
        count += 1
        for k2, filt2 in enumerate(scattering.psi2_f):
            if filt2['j'] > filt1['j']:
                assert (k1, k2) in s.keys()
                count += 1

    assert count == len(s)

    with pytest.raises(ValueError) as ve:
        scattering.vectorize = True
        scattering.forward(x)
    assert "mutually incompatible" in ve.value.args[0]


# Technical tests
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_scattering_GPU_CPU(device, backend, random_state=42, test_cuda=None):
    """
    This function tests whether the CPU computations are equivalent to
    the GPU ones

    Note that we can only achieve 1e-4 absolute l_infty error between GPU
    and CPU
    """
    if device=='cpu':
        pytest.skip("Only for GPU use...")

    torch.manual_seed(random_state)

    J = 6
    Q = 8
    T = 2**12


    # build the scattering
    scattering = Scattering1D(J, T, Q, backend=backend, frontend='torch')

    x = torch.randn(128, T)
    s_cpu = scattering.forward(x)

    scattering = scattering.cuda()
    x_gpu = x.clone().cuda()
    s_gpu = scattering.forward(x_gpu).cpu()
    # compute the distance
    assert torch.max(torch.abs(s_cpu - s_gpu)) < 1e-4

@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_coordinates(device, backend, random_state=42):
    """
    Tests whether the coordinates correspond to the actual values (obtained
    with Scattering1d.meta()), and with the vectorization
    """
    torch.manual_seed(random_state)
    J = 6
    Q = 8
    T = 2**12
    scattering = Scattering1D(J, T, Q, max_order=2, backend=backend, frontend='torch')
    x = torch.randn(128, T)


    scattering.to(device)
    x = x.to(device)

    for max_order in [1, 2]:
        scattering.max_order = max_order

        scattering.vectorize = False
        s_dico = scattering.forward(x)
        s_dico = {k: s_dico[k].data for k in s_dico.keys()}
        scattering.vectorize = True
        s_vec = scattering.forward(x)


        s_dico = {k: s_dico[k].cpu() for k in s_dico.keys()}
        s_vec = s_vec.cpu()

        meta = scattering.meta()

        assert len(s_dico) == s_vec.shape[1]

        for cc in range(s_vec.shape[1]):
            k = meta['key'][cc]
            diff = s_vec[:, cc] - torch.squeeze(s_dico[k])
            assert torch.max(torch.abs(diff)) < 1e-7


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_precompute_size_scattering(device, backend, random_state=42):
    """
    Tests that precompute_size_scattering computes a size which corresponds
    to the actual scattering computed
    """
    torch.manual_seed(random_state)
    J = 6
    Q = 8
    T = 2**12
    scattering = Scattering1D(J, T, Q, vectorize=False, backend=backend, frontend='torch')
    x = torch.randn(128, T)


    scattering.to(device)
    x = x.to(device)

    for max_order in [1, 2]:
        scattering.max_order = max_order
        s_dico = scattering.forward(x)
        for detail in [True, False]:
            # get the size of scattering
            size = scattering.output_size(detail=detail)
            if detail:
                num_orders = {0: 0, 1: 0, 2: 0}
                for k in s_dico.keys():
                    if k is ():
                        num_orders[0] += 1
                    else:
                        if len(k) == 1:  # order1
                            num_orders[1] += 1
                        elif len(k) == 2:
                            num_orders[2] += 1
                todo = 2 if max_order == 2 else 1
                for i in range(todo):
                    assert num_orders[i] == size[i]
                    # check that the orders are completely equal
            else:
                assert len(s_dico) == size

@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_differentiability_scattering(device, backend, random_state=42):
    """
    It simply tests whether it is really differentiable or not.
    This does NOT test whether the gradients are correct.
    """

    if backend.name == "torch_skcuda":
        pytest.skip("The skcuda backend does not pass differentiability"
            "tests, but that's ok (for now).")

    torch.manual_seed(random_state)
    J = 6
    Q = 8
    T = 2**12
    scattering = Scattering1D(J, T, Q, frontend='torch', backend=backend).to(device)
    x = torch.randn(128, T, requires_grad=True).to(device)

    s = scattering.forward(x)
    loss = torch.sum(torch.abs(s))
    loss.backward()
    assert torch.max(torch.abs(x.grad)) > 0.


@pytest.mark.parametrize("backend", backends)
def test_scattering_shape_input(backend):
    # Checks that a wrong input to shape raises an error
    J, Q = 6, 8
    with pytest.raises(ValueError) as ve:
        shape = 5, 6
        s = Scattering1D(J, shape, Q, backend=backend, frontend='torch')
    assert "exactly one element" in ve.value.args[0]


    with pytest.raises(ValueError) as ve:
        shape = 1.5
        s = Scattering1D(J, shape, Q, backend=backend, frontend='torch')
        # should invoke the else branch
    assert "1-tuple" in ve.value.args[0]
    assert "integer" in ve.value.args[0]

@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_batch_shape_agnostic(device, backend):
    J, Q = 3, 8
    length = 1024
    shape = (length,)

    length_ds = length / 2**J

    S = Scattering1D(J, shape, Q, backend=backend, frontend='torch').to(device)

    with pytest.raises(ValueError) as ve:
        S(torch.zeros(()).to(device))
    assert "at least one axis" in ve.value.args[0]

    x = torch.zeros(shape).to(device)

    Sx = S(x)

    assert Sx.dim() == 2
    assert Sx.shape[-1] == length_ds

    n_coeffs = Sx.shape[-2]

    test_shapes = ((1,) + shape, (2,) + shape, (2,2) + shape, (2,2,2) + shape)

    for test_shape in test_shapes:
        x = torch.zeros(test_shape).to(device)

        S.vectorize = True
        Sx = S(x)

        assert Sx.dim() == len(test_shape)+1
        assert Sx.shape[-1] == length_ds
        assert Sx.shape[-2] == n_coeffs
        assert Sx.shape[:-2] == test_shape[:-1]

        S.vectorize = False
        Sx = S(x)

        assert len(Sx) == n_coeffs
        for k, v in Sx.items():
            assert v.shape[-1] == length_ds
            assert v.shape[-2] == 1
            assert v.shape[:-2] == test_shape[:-1]

@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_pad_1d(device, backend, random_state=42):
    """
    Tests the correctness and differentiability of pad_1d
    """
    torch.manual_seed(random_state)
    N = 128
    for pad_left in range(0, N, 16):
        for pad_right in range(0, N, 16):
            x = torch.randn(100, 4, N, requires_grad=True).to(device)
            x_pad = backend.pad_1d(x, pad_left, pad_right, mode='reflect')
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
        backend.pad_1d(x, x.shape[-1], 0, mode='reflect')
    with pytest.raises(ValueError):
        backend.pad_1d(x, 0, x.shape[-1], mode='reflect')

@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_modulus(device, backend, random_state=42):
    """
    Tests the stability and differentiability of modulus
    """
    torch.manual_seed(random_state)
    # Test with a random vector
    x = torch.randn(100, 4, 128, 2, requires_grad=True).to(device)
    x_abs = backend.modulus_complex(x)

    x_abs = x_abs.to(device)
    assert len(x_abs.shape) == len(x.shape)
    # check the value
    x_abs2 = x_abs.clone()
    x2 = x.clone()

    diff = x_abs2[..., 0] - torch.sqrt(x2[..., 0]**2 + x2[..., 1]**2)
    assert torch.max(torch.abs(diff)) <= 1e-6

    # If we are using a GPU-only backend, make sure it raises the proper
    # errors for CPU tensors.
    if backend.name == 'torch_skcuda':
        with pytest.raises(RuntimeError) as re:
            x_bad = torch.randn((4, 2)).cpu()
            backend.modulus_complex(x_bad)
        assert "for cpu tensors" in re.value.args[0].lower()

    with pytest.raises(TypeError) as te:
        x_bad = torch.randn(4).to(device)
        backend.modulus_complex(x_bad)
    assert "should be complex" in te.value.args[0]

    if backend.name == "torch_skcuda":
        pytest.skip("The skcuda backend does not pass differentiability"
            "tests, but that's ok (for now).")

    # check the gradient
    loss = torch.sum(x_abs)
    loss.backward()
    x_grad = x2 / x_abs2[..., 0].unsqueeze(dim=-1)
    diff = x.grad - x_grad
    assert torch.max(torch.abs(diff)) <= 1e-7

    # Manually check `forward`/`backward` using fake context object. This
    # ensures that `backward` is included in code coverage since going through
    # the PyTorch C extension seems to not play well with `coverage.py`.
    class FakeContext:
        def save_for_backward(self, *args):
            self.saved_tensors = args

    ctx = FakeContext()
    y = backend.ModulusStable.forward(ctx, x)
    y_grad = torch.ones_like(y)
    x_grad_manual = backend.ModulusStable.backward(ctx, y_grad)
    assert (x_grad_manual - x_grad).abs().max() < 1e-7

    # Test the differentiation with a vector made of zeros
    x0 = torch.zeros(100, 4, 128, 2, requires_grad=True).to(device)
    x_abs0 = backend.modulus_complex(x0)
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
    rng = np.random.RandomState(random_state)
    J = 10
    x = rng.randn(100, 4, 2**J) + 1j * rng.randn(100, 4, 2**J)
    x_f = np.fft.fft(x, axis=-1)[..., np.newaxis]
    x_f.dtype = 'float64'  # make it a vector
    x_f_th = torch.from_numpy(x_f)

    for j in range(J + 1):
        x_f_sub_th = backend.subsample_fourier(x_f_th, 2**j).cpu()
        x_f_sub = x_f_sub_th.numpy()
        x_f_sub.dtype = 'complex128'
        x_sub = np.fft.ifft(x_f_sub[..., 0], axis=-1)
        assert np.max(np.abs(x[:, :, ::2**j] - x_sub)) < 1e-7

    # If we are using a GPU-only backend, make sure it raises the proper
    # errors for CPU tensors.
    if backend.name == 'torch_skcuda':
        with pytest.raises(RuntimeError) as re:
            x_bad = torch.randn((4, 2)).cpu()
            backend.subsample_fourier(x_bad, 1)
        assert "for cpu tensors" in re.value.args[0].lower()

    if device=='cuda':
        with pytest.raises(TypeError) as te:
            x_bad = torch.randn(4).cuda()
            backend.subsample_fourier(x_bad, 1)
        assert "should be complex" in te.value.args[0]


