import pytest
import torch
from kymatio import Scattering1D
import math
import os
import io
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
def test_simple_scatterings(device, backend, random_state=42):
    """
    Checks the behaviour of the scattering on simple signals
    (zero, constant, pure cosine)
    """

    rng = np.random.RandomState(random_state)
    J = 6
    Q = 8
    T = 2**9
    scattering = Scattering1D(J, T, Q, backend=backend, frontend='torch').to(device)
    return

    # zero signal
    x0 = torch.zeros(2, T).to(device)

    if backend.name.endswith('_skcuda') and device == 'cpu':
        with pytest.raises(TypeError) as ve:
            s = scattering(x0)
        assert "CPU" in ve.value.args[0]
        return
    s = scattering(x0)

    # check that s is zero!
    assert torch.max(torch.abs(s)) < 1e-7

    # constant signal
    x1 = rng.randn(1)[0] * torch.ones(1, T).to(device)
    if not backend.name.endswith('_skcuda') or device != 'cpu':
        s1 = scattering(x1)

        # check that all orders above 1 are 0
        assert torch.max(torch.abs(s1[:, 1:])) < 1e-7

    # sinusoid scattering
    meta = scattering.meta()
    for _ in range(3):
        k = rng.randint(1, T // 2, 1)[0]
        x2 = torch.cos(2 * math.pi * float(k) * torch.arange(0, T, dtype=torch.float32) / float(T))
        x2 = x2.unsqueeze(0).to(device)
        if not backend.name.endswith('_skcuda') or device != 'cpu':
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

    with open(os.path.join(test_data_dir, 'test_data_1d.npz'), 'rb') as f:
        buffer = io.BytesIO(f.read())
        data = np.load(buffer)


    x = torch.from_numpy(data['x']).to(device)
    J = int(data['J'])
    Q = int(data['Q'])
    Sx0 = torch.from_numpy(data['Sx']).to(device)

    T = x.shape[-1]

    scattering = Scattering1D(J, T, Q, backend=backend, frontend='torch').to(device)

    if backend.name.endswith('_skcuda') and device == 'cpu':
        with pytest.raises(TypeError) as ve:
            Sx = scattering(x)
        assert "CUDA" in ve.value.args[0]
        return

    Sx = scattering(x)

    assert Sx.shape == Sx0.shape
    assert torch.allclose(Sx, Sx0)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_computation_Ux(backend, device, random_state=42):
    """
    Checks the computation of the U transform (no averaging for 1st order)
    """
    rng = np.random.RandomState(random_state)
    J = 6
    Q = 8
    shape = 2**12
    scattering = Scattering1D(J, shape, Q, T=0, max_order=1, out_type="dict",
        frontend='torch', backend=backend).to(device)
    # random signal
    x = torch.from_numpy(rng.randn(1, shape)).float().to(device)

    if not backend.name.endswith('skcuda') or device != 'cpu':
        s = scattering(x)

        # check that the keys in s correspond to the order 0 and second order
        for k in range(len(scattering.psi1_f)):
            assert (k,) in s.keys()
        for k in s.keys():
            if k != ():
                assert k[0] < len(scattering.psi1_f)

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
            scattering.out_type = "array"
            scattering(x)
        assert "Please set out_type to" in ve.value.args[0]


# Technical tests
@pytest.mark.parametrize("backend", backends)
def test_scattering_GPU_CPU(backend, random_state=42):
    """
    This function tests whether the CPU computations are equivalent to
    the GPU ones
    """
    if torch.cuda.is_available() and not backend.name.endswith('_skcuda'):
        torch.manual_seed(random_state)

        J = 6
        Q = 8
        T = 2**12

        # build the scattering
        scattering = Scattering1D(J, T, Q, backend=backend, frontend='torch').cpu()

        x = torch.randn(2, T)
        s_cpu = scattering(x)

        scattering = scattering.cuda()
        x_gpu = x.clone().cuda()
        s_gpu = scattering(x_gpu).cpu()
        # compute the distance

        Warning('Tolerance has been slightly lowered here...')
        assert torch.allclose(s_cpu, s_gpu, atol=1e-7)


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

    x = torch.randn(2, T)

    scattering.to(device)
    x = x.to(device)

    for max_order in [1, 2]:
        scattering.max_order = max_order

        scattering.out_type = "dict"

        if backend.name.endswith('skcuda') and device == 'cpu':
            with pytest.raises(TypeError) as ve:
                s_dico = scattering(x)
            assert "CUDA" in ve.value.args[0]
        else:
            s_dico = scattering(x)
            s_dico = {k: s_dico[k].data for k in s_dico.keys()}
        scattering.out_type = "array"

        if backend.name.endswith('_skcuda') and device == 'cpu':
            with pytest.raises(TypeError) as ve:
                s_vec = scattering(x)
            assert "CUDA" in ve.value.args[0]
        else:
            s_vec = scattering(x)
            s_dico = {k: s_dico[k].cpu() for k in s_dico.keys()}
            s_vec = s_vec.cpu()

        meta = scattering.meta()

        if not backend.name.endswith('_skcuda') or device != 'cpu':
            assert len(s_dico) == s_vec.shape[1]

            for cc in range(s_vec.shape[1]):
                k = meta['key'][cc]
                assert torch.allclose(s_vec[:, cc], torch.squeeze(s_dico[k]))


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_output_size(device, backend, random_state=42):
    """
    Tests that S.output_size() returns the same size as S.scattering(x)
    """
    torch.manual_seed(random_state)

    J = 6
    Q = 8
    N = 2**12

    scattering = Scattering1D(J, N, Q, out_type="dict", backend=backend, frontend='torch')

    x = torch.randn(2, N)

    scattering.to(device)
    x = x.to(device)
    if not backend.name.endswith('_skcuda') or device != 'cpu':
        for max_order in [1, 2]:
            scattering.max_order = max_order
            s_dico = scattering(x)
            for detail in [True, False]:
                # get the size of scattering
                size = scattering.output_size(detail=detail)
                if detail:
                    num_orders = {0: 0, 1: 0, 2: 0}
                    for k in s_dico.keys():
                        if k == ():
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

    if backend.name.endswith("_skcuda"):
        pytest.skip("The skcuda backend does not pass differentiability"
            "tests, but that's ok (for now).")

    torch.manual_seed(random_state)

    J = 6
    Q = 8
    T = 2**12

    scattering = Scattering1D(J, T, Q, frontend='torch', backend=backend).to(device)

    x = torch.randn(2, T, requires_grad=True, device=device)

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

    if backend.name.endswith('_skcuda') and device == 'cpu':
        with pytest.raises(TypeError) as ve:
            Sx = S(x)
        assert "CUDA" in ve.value.args[0]
        return

    Sx = S(x)

    assert Sx.dim() == 2
    assert Sx.shape[-1] == length_ds

    n_coeffs = Sx.shape[-2]

    test_shapes = ((1,) + shape, (2,) + shape, (2,2) + shape, (2,2,2) + shape)

    for test_shape in test_shapes:
        x = torch.zeros(test_shape).to(device)

        S.out_type = "array"
        Sx = S(x)

        assert Sx.dim() == len(test_shape)+1
        assert Sx.shape[-1] == length_ds
        assert Sx.shape[-2] == n_coeffs
        assert Sx.shape[:-2] == test_shape[:-1]

        S.out_type = "dict"
        Sx = S(x)

        assert len(Sx) == n_coeffs
        for k, v in Sx.items():
            assert v.shape[-1] == length_ds
            assert v.shape[:-1] == test_shape[:-1]


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_T(device, backend):
    """
    Applies scattering on a stored signal with non-default T to make sure its
    output agrees with a previously calculated version.
    """
    test_data_dir = os.path.dirname(__file__)

    with open(os.path.join(test_data_dir, 'test_data_1d.npz'), 'rb') as f:
        buffer = io.BytesIO(f.read())
        data = np.load(buffer)


    x = torch.from_numpy(data['x']).to(device)
    J = int(data['J'])
    Q = int(data['Q'])
    Sx0 = torch.from_numpy(data['Sx']).to(device)

    # default
    N = x.shape[-1]
    T = 2**J
    scattering0 = Scattering1D(J, N, Q, T=T, backend=backend, frontend='torch'
                               ).to(device)

    # adjust T
    sigma_low_scale_factor = 2
    T = 2**(J - sigma_low_scale_factor)
    scattering1 = Scattering1D(J, N, Q, T=T, backend=backend, frontend='torch'
                               ).to(device)

    if backend.name.endswith('_skcuda') and device == 'cpu':
        for scattering in (scattering0, scattering1):
            with pytest.raises(TypeError) as ve:
                _ = scattering(x)
            assert "CUDA" in ve.value.args[0]
        return

    Sg0 = scattering0(x)
    Sg1 = scattering1(x)
    assert torch.allclose(Sg0, Sx0)
    assert Sg1.shape == (Sg0.shape[0], Sg0.shape[1],
        Sg0.shape[2]*(2**sigma_low_scale_factor))

    # adjust oversampling
    oversampling = 1
    scattering2 = Scattering1D(J, N, Q, backend=backend,
        frontend='torch', oversampling=oversampling).to(device)
    Sg2 = scattering2(x)
    assert Sg2.shape == (Sg0.shape[0], Sg0.shape[1],
        Sg0.shape[2]*(2**oversampling))


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_Q(device, backend):
    J = 3
    length = 1024
    shape = (length,)

    # test different cases for Q
    with pytest.raises(ValueError) as ve:
        S = Scattering1D(
            J, shape, Q=0.9, backend=backend, frontend='torch')
        Q = S.Q
    assert "Q must always be >= 1" in ve.value.args[0]

    with pytest.raises(ValueError) as ve:
        S = Scattering1D(
            J, shape, Q=[8], backend=backend, frontend='torch')
        Q = S.Q
    assert "Q must be an integer or a tuple" in ve.value.args[0]

    Sc_int = Scattering1D(J, shape, Q=(8, ), backend=backend, frontend='torch').to(device)
    Sc_tuple = Scattering1D(J, shape, Q=(8, 1), backend=backend, frontend='torch').to(device)

    assert Sc_int.Q == Sc_tuple.Q

    # test dummy input
    x = torch.zeros(shape).to(device)

    if backend.name.endswith('_skcuda') and device == 'cpu':
        for scattering in (Sc_int, Sc_tuple):
            with pytest.raises(TypeError) as ve:
                _ = scattering(x)
            assert "CUDA" in ve.value.args[0]
        return

    Sc_int_out = Sc_int(x)
    Sc_tuple_out = Sc_tuple(x)

    assert torch.allclose(Sc_int_out, Sc_tuple_out)
    assert Sc_int_out.shape == Sc_tuple_out.shape


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("backend", backends)
def test_check_runtime_args(device, backend):
    J = 3
    length = 1024
    shape = (length,)
    x = torch.zeros(shape)

    with pytest.raises(ValueError) as ve:
        S = Scattering1D(J, shape, backend=backend,
                         out_type='doesnotexist', frontend='torch').to(device)
        S(x)
    assert "out_type must be one" in ve.value.args[0]

    with pytest.raises(ValueError) as ve:
        S = Scattering1D(J, shape, backend=backend, T=0,
                         out_type='array', frontend='torch').to(device)
        S(x)
    assert "Cannot convert" in ve.value.args[0]

    with pytest.raises(ValueError) as ve:
        S = Scattering1D(J, shape, oversampling=-1, backend=backend,
                         frontend='torch').to(device)
        S(x)
    assert "nonnegative" in ve.value.args[0]

    with pytest.raises(ValueError) as ve:
        S = Scattering1D(J, shape, oversampling=0.5, backend=backend,
                         frontend='torch').to(device)
        S(x)
    assert "integer" in ve.value.args[0]

    with pytest.raises(ValueError) as ve:
        S = Scattering1D(J, shape, stride="noninteger",
            backend=backend, frontend='torch').to(device)
        S(x)
    assert "integer" in ve.value.args[0]

    with pytest.raises(ValueError) as ve:
        S = Scattering1D(J, shape, stride=17,
            backend=backend, frontend='torch').to(device)
        S(x)
    assert "power of two" in ve.value.args[0]

    with pytest.raises(ValueError) as ve:
        S = Scattering1D(J, shape, T=0, stride=16, out_type="list",
            backend=backend, frontend='torch').to(device)
        S(x)
    assert "incompatible" in ve.value.args[0]

    with pytest.raises(ValueError) as ve:
        S = Scattering1D(J, shape, T="global", stride=16,
            backend=backend, frontend='torch').to(device)
        S(x)
    assert "incompatible" in ve.value.args[0]

    with pytest.raises(ValueError) as ve:
        S = Scattering1D(J, shape, stride=16, oversampling=1,
            backend=backend, frontend='torch').to(device)
        S(x)
    assert "incompatible" in ve.value.args[0]

def test_Scattering1D_average_global():
    """
    Tests global averaging.
    """
    N = 2 ** 13
    Q = (1, 1)
    J = 5
    T = 'global'
    sc = Scattering1D(J, N, Q, T, backend=backends[-1], frontend='torch', out_type='array')
    x = torch.zeros((N,))
    Sx = sc(x)
    assert Sx.shape[-1] == 1
