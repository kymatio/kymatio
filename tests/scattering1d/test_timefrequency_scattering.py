import numpy as np
import pytest
import torch, tensorflow as tf
import jax.numpy as jnp
from jax import device_put

import kymatio
from kymatio import TimeFrequencyScattering
from kymatio.scattering1d.core.scattering1d import scattering1d
from kymatio.scattering1d.filter_bank import gauss_1d
from kymatio.scattering1d.core.timefrequency_scattering import (
    joint_timefrequency_scattering,
    time_scattering_widthfirst,
    frequency_scattering,
    time_averaging,
    frequency_averaging,
)
from kymatio.scattering1d.frontend.base_frontend import TimeFrequencyScatteringBase
from kymatio.scattering1d.frontend.torch_frontend import TimeFrequencyScatteringTorch
from kymatio.scattering1d.frontend.numpy_frontend import TimeFrequencyScatteringNumPy
from kymatio.jax import TimeFrequencyScattering as TimeFrequencyScatteringJax


backends = ["numpy", "torch", "tensorflow", "jax", "sklearn"]
@pytest.mark.parametrize("backend", backends)
def test_jtfs_build(backend):
    # Test __init__
    jtfs = TimeFrequencyScatteringBase(J=10, J_fr=3, shape=4096, Q=8, backend=backend)
    assert jtfs.F == (2**jtfs.J_fr)

    # Test Q_fr
    jtfs.build()
    assert jtfs.Q_fr == (1,)

    # Test parse_T(self.F, self.J_fr, N_input_fr, T_alias='F')
    assert jtfs.F == (2**jtfs.J_fr)
    assert jtfs.log2_F == jtfs.J_fr

    # Test that frequential padding is sufficient
    N_input_fr = len(jtfs.psi1_f)
    assert (jtfs._N_padded_fr - N_input_fr) > (2 * jtfs.F)
    assert (jtfs._N_padded_fr - N_input_fr) > (2 * (2**jtfs.J_fr))
    phi_f = gauss_1d(2 * jtfs._N_padded_fr, jtfs.sigma0 / jtfs.F)
    h_double_support = np.fft.fftshift(np.fft.ifft(phi_f))
    h_current_support = h_double_support[
        (jtfs._N_padded_fr // 2) : -jtfs._N_padded_fr // 2
    ]
    l1_current = np.linalg.norm(h_current_support, ord=1)
    l1_double = np.linalg.norm(h_double_support, ord=1)
    l1_residual = l1_double - l1_current
    criterion_amplitude = 1e-3
    assert l1_residual < criterion_amplitude

    # Test that padded frequency domain is divisible by max subsampling factor
    assert (jtfs._N_padded_fr % (2**jtfs.J_fr)) == 0


backends = ["numpy", "torch", "tensorflow", "jax", "sklearn"]
@pytest.mark.parametrize("backend", backends)
def test_Q(backend):
    jtfs_kwargs = dict(J=10, J_fr=3, shape=4096, Q=8, backend=backend)
    # test different cases for Q_fr
    with pytest.raises(ValueError) as ve:
        TimeFrequencyScatteringBase(Q_fr=0.9, **jtfs_kwargs).build()
    assert "Q_fr must be >= 1" in ve.value.args[0]

    # test different cases for Q_fr
    with pytest.raises(ValueError) as ve:
        TimeFrequencyScatteringBase(Q_fr=[1], **jtfs_kwargs).build()
    assert "Q_fr must be an integer or 1-tuple" in ve.value.args[0]

    with pytest.raises(NotImplementedError) as ve:
        TimeFrequencyScatteringBase(Q_fr=(1, 2), **jtfs_kwargs).build()
    assert "Q_fr must be an integer or 1-tuple" in ve.value.args[0]


def test_check_runtime_args():
    kwargs = dict(J=10, J_fr=3, shape=4096, Q=8)
    S = TimeFrequencyScatteringBase(**kwargs)
    S.build()
    assert S._check_runtime_args() is None

    with pytest.raises(ValueError) as ve:
        S = TimeFrequencyScatteringBase(out_type="doesnotexist", **kwargs)
        S.build()
        S._check_runtime_args()
    assert "out_type must be one of" in ve.value.args[0]

    with pytest.raises(ValueError) as ve:
        S = TimeFrequencyScatteringBase(T=0, **kwargs)
        S.build()
        S._check_runtime_args()
    assert "Cannot convert" in ve.value.args[0]

    with pytest.raises(ValueError) as ve:
        S = TimeFrequencyScatteringBase(F=0, out_type="array", format="joint", **kwargs)
        S.build()
        S._check_runtime_args()
    assert "Cannot convert" in ve.value.args[0]

    with pytest.raises(ValueError) as ve:
        S = TimeFrequencyScatteringBase(format="doesnotexist", **kwargs)
        S.build()
        S._check_runtime_args()
    assert "format must be" in ve.value.args[0]


backends = ["numpy", "torch", "tensorflow", "jax", "sklearn"]
@pytest.mark.parametrize("backend", backends)
def test_jtfs_create_filters(backend):
    J_fr = 3
    jtfs = TimeFrequencyScatteringBase(
        J=10, J_fr=J_fr, shape=4096, Q=8, backend=backend
    )
    jtfs.build()
    jtfs.create_filters()

    phi = jtfs.filters_fr[0]
    assert phi["N"] == jtfs._N_padded_fr

    psis = jtfs.filters_fr[1]
    assert len(psis) == (1 + 2 * (J_fr + 1))
    assert np.isclose(psis[0]["xi"], 0)
    positive_spins = psis[1:][: J_fr + 1]
    negative_spins = psis[1:][J_fr + 1 :]
    assert all([psi["xi"] > 0 for psi in positive_spins])
    assert all([psi["xi"] < 0 for psi in negative_spins])
    assert all(
        [
            (psi_pos["xi"] == -psi_neg["xi"])
            for psi_pos, psi_neg in zip(positive_spins, negative_spins)
        ]
    )


frontends = ["numpy", "torch", "tensorflow", "sklearn", "jax"]
@pytest.mark.parametrize("frontend", frontends)
def test_time_scattering_widthfirst(frontend):
    """Checks that width-first and depth-first algorithms have same output."""
    J = 5
    shape = (1024,)
    S = kymatio.Scattering1D(J, shape, frontend=frontend)
    x = torch.zeros(shape) if frontend == "torch" else np.zeros(shape)
    x[shape[0] // 2] = 1
    x_shape = S.backend.shape(x)
    _, signal_shape = x_shape[:-1], x_shape[-1:]
    x = S.backend.reshape_input(x, signal_shape)
    U_0 = S.backend.pad(x, pad_left=S.pad_left, pad_right=S.pad_right)

    # Width-first
    filters = [S.phi_f, S.psi1_f, S.psi2_f]
    W_gen = time_scattering_widthfirst(
        U_0, S.backend, filters, S.log2_stride, average_local=True
    )
    W = {path["n"]: path["coef"] for path in W_gen}

    # Depth-first
    S_gen = scattering1d(U_0, S.backend, filters, S.log2_stride, average_local=True)
    S1_depth = {path["n"]: path["coef"] for path in S_gen if len(path["n"]) > 0}

    # Check order 1
    keep_order1 = lambda item: (len(item[0]) == 1)
    S1_width = dict(filter(keep_order1, W.items()))
    assert len(S1_width) == 1
    S1_width = S1_width[(-1,)]
    S1_depth = S.backend.stack(
        [S1_depth[key] for key in sorted(S1_depth.keys()) if len(key) == 1], 2
    )
    if frontend not in ["numpy", "sklearn", "jax"]:
        S1_width = S1_width.numpy()
        S1_depth = S1_depth.numpy()
    assert np.allclose(S1_width, S1_depth)


frontends = ["numpy"]
@pytest.mark.parametrize("frontend", frontends)
def test_frequency_scattering(frontend):
    # Create S1 array as input
    J = 5
    shape = (4096,)
    Q = 8
    S = kymatio.Scattering1D(J=J, Q=Q, shape=shape, frontend=frontend)
    x = np.zeros(shape)
    x[shape[0] // 2] = 1
    x_shape = S.backend.shape(x)
    _, signal_shape = x_shape[:-1], x_shape[-1:]
    x = S.backend.reshape_input(x, signal_shape)
    U_0 = S.backend.pad(x, pad_left=S.pad_left, pad_right=S.pad_right)
    filters = [S.phi_f, S.psi1_f, S.psi2_f]
    average_local = True
    S_gen = scattering1d(U_0, S.backend, filters, S.log2_stride, average_local)
    S_1_dict = {path["n"]: path["coef"] for path in S_gen if len(path["n"]) == 1}
    S_1 = S.backend.stack([S_1_dict[key] for key in sorted(S_1_dict.keys())], 2)
    X = {"coef": S_1, "n1_max": len(S_1_dict), "n": (-1,), "j": (-1,)}

    # Define scattering object
    J_fr = 3
    jtfs = TimeFrequencyScatteringBase(
        J=J, J_fr=J_fr, shape=shape, Q=Q, backend=frontend
    )
    jtfs.build()
    jtfs.create_filters()

    # spinned=False
    freq_gen = frequency_scattering(
        X,
        S.backend,
        jtfs.filters_fr,
        jtfs.log2_stride_fr,
        jtfs.average_fr == "local",
        spinned=False,
    )
    for Y_fr in freq_gen:
        assert Y_fr["spin"] >= 0
        assert Y_fr["n"] == Y_fr["n_fr"]

    # spinned=True
    X["coef"] = X["coef"].astype("complex64")
    X["n"] = (-1, 4)  # a mockup (n1, n2) pair from time_scattering_widthfirst
    freq_gen = frequency_scattering(
        X,
        S.backend,
        jtfs.filters_fr,
        jtfs.log2_stride_fr,
        jtfs.average_fr == "local",
        spinned=True,
    )
    for Y_fr in freq_gen:
        assert Y_fr["coef"].shape[-1] == X["coef"].shape[-1]
        assert Y_fr["n"] == (4,) + Y_fr["n_fr"]


def _jtfs_test_routine(S, backend, shape):
    x = torch.zeros(shape)
    x[shape[0] // 2] = 1
    x_shape = backend.shape(x)
    _, signal_shape = x_shape[:-1], x_shape[-1:]
    x = backend.reshape_input(x, signal_shape)
    U_0_in = backend.pad(x, pad_left=S.pad_left, pad_right=S.pad_right)

    filters = [S.phi_f, S.psi1_f, S.psi2_f]
    jtfs_gen = joint_timefrequency_scattering(
        U_0_in,
        backend,
        filters,
        S.log2_stride,
        S.average == "local",
        S.filters_fr,
        S.log2_stride_fr,
        S.average_fr == "local",
    )
    path_keys = ["coef", "j", "n", "n1_max", "n1_stride", "j_fr", "n_fr", "spin"]

    # Zeroth order
    U_0 = next(jtfs_gen)
    if S.average:
        assert (U_0["coef"].shape[-1]*S.stride) == U_0_in.shape[-1]
    else:
        assert U_0["coef"].shape == U_0_in.shape
        assert np.allclose(U_0["coef"], U_0_in)
    assert U_0["n"] == ()
    assert U_0["j"] == ()

    # First and second order
    S_jtfs = list(jtfs_gen)

    # Check that all path keys are unique
    ns = [path["n"] for path in S_jtfs]
    assert len(ns) == len(set(ns))

    # Check that order 1 comes before order 2
    orders = [len(path["n"]) for path in S_jtfs]
    assert set(orders) == {1, 2}
    assert orders == sorted(orders)

    # Test first order
    S1_jtfs = filter(lambda path: len(path["n"]) == 1, S_jtfs)
    for path in S1_jtfs:
        # Check path format
        assert path.keys() == set(path_keys)

        # Check that first-order spins are nonnegative
        assert path["spin"] >= 0

        # Check frequency padding
        assert path["n1_max"] < S._N_padded_fr

        # Check that first-order coefficients have the same temporal stride
        assert (path["coef"].shape[-1] * S.stride) == S._N_padded

        # Check that frequential stride works as intended
        stride_fr = (2 ** path["j_fr"][0])
        assert (path["coef"].shape[-2] * stride_fr) == S._N_padded_fr

        # Check that padding is sufficient
        midpoint = (path["n1_max"] + S._N_padded_fr) // (2 * stride_fr)
        avg_value = np.mean(np.abs(path["coef"][..., midpoint, :]))
        assert avg_value < 1e-5

        # Check that averaged coefficients have the right temporal stride
        if S.average == "local":
            assert (path["coef"].shape[-1] * S.stride) == S._N_padded

    # Test second order
    S2_jtfs = filter(lambda path: len(path["n"]) == 2, S_jtfs)
    for path in S2_jtfs:
        # Check path format
        assert path.keys() == set(path_keys)

        # Check frequency padding
        assert path["n1_max"] < S._N_padded_fr

        # Check that temporal stride works as intended
        assert (path["coef"].shape[-1] * (2**path["j"][1])) == S._N_padded

        # Check that frequential stride works as intended
        stride_fr = (2 ** path["j_fr"][0])
        assert (path["coef"].shape[-2] * stride_fr) == S._N_padded_fr

        # Check that padding is sufficient
        midpoint = (path["n1_max"] + S._N_padded_fr) // (2 * stride_fr)
        avg_value = np.mean(np.abs(path["coef"][..., midpoint, :]))
        assert avg_value < 1e-5

    # Test frequential averaging
    U_2 = {**path, "coef": backend.modulus(path["coef"])}

    # average_fr == 'local'
    average_fr = "local"
    S_2 = frequency_averaging(
        U_2, backend, S.filters_fr[0], S.log2_stride_fr, average_fr
    )
    stride_fr = (2 ** S.log2_stride_fr)
    assert S_2["n1_stride"] == stride_fr
    assert (S_2["coef"].shape[-2] * stride_fr) == S._N_padded_fr

    # average_fr == 'global'
    average_fr = "global"
    S_2 = frequency_averaging(
        U_2, backend, S.filters_fr[0], S.log2_stride_fr, average_fr
    )
    assert S_2["n1_stride"] == S_2["n1_max"]
    assert S_2["coef"].shape[-2] == 1

    # average_fr == False
    average_fr = False
    S_2 = frequency_averaging(
        U_2, backend, S.filters_fr[0], S.log2_stride_fr, average_fr
    )
    stride_fr = (2 ** S_2["j_fr"][0])
    assert S_2["n1_stride"] == stride_fr
    assert np.allclose(S_2["coef"], U_2["coef"])

    # Check that second-order spins are mirrors of each other
    for path in S2_jtfs:
        del path["coef"]
    S2_pos = filter(lambda path: path["spin"] > 0, S2_jtfs)
    S2_neg = filter(lambda path: path["spin"] < 0, S2_jtfs)
    assert len(list(S2_pos)) == len(list(S2_neg))
    assert set(S2_pos) == set(S2_neg)


backends = ["numpy", "tensorflow"]
@pytest.mark.parametrize("backend", backends)
def test_joint_timefrequency_scattering(backend):
    # Define scattering object
    J = 8
    J_fr = 3
    shape = (4096,)
    Q = 8

    # no average, no average_fr
    S = TimeFrequencyScatteringBase(
        J=J, J_fr=J_fr, shape=shape, Q=Q, T=0, F=0, backend=backend
    )
    S.build()
    S.create_filters()
    assert not S.average
    assert not S.average_fr
    sc1d_backend = kymatio.Scattering1D(
        J=J, Q=Q, shape=shape, T=0, frontend=backend).backend
    _jtfs_test_routine(S, sc1d_backend, shape)

    # average, average_fr
    S = TimeFrequencyScatteringBase(
        J=J, J_fr=J_fr, shape=shape, Q=Q, backend=backend
    )
    S.build()
    S.create_filters()
    assert S.average
    assert S.average_fr
    sc1d_backend = kymatio.Scattering1D(
        J=J, Q=Q, shape=shape, frontend=backend).backend
    _jtfs_test_routine(S, sc1d_backend, shape)


def test_differentiability_jtfs_torch(random_state=42):
    device = "cpu"
    J = 8
    J_fr = 3
    shape = (4096,)
    Q = 8
    S = TimeFrequencyScatteringTorch(J=J, J_fr=J_fr, shape=shape, Q=Q, T=0, F=0).to(
        device
    )
    S.build()
    S.create_filters()
    S.load_filters()
    backend = S.backend
    torch.manual_seed(random_state)

    x = torch.randn(shape, requires_grad=True, device=device)
    x_shape = backend.shape(x)
    _, signal_shape = x_shape[:-1], x_shape[-1:]
    x_reshaped = backend.reshape_input(x, signal_shape)
    U_0_in = backend.pad(x_reshaped, pad_left=S.pad_left, pad_right=S.pad_right)

    filters = [S.phi_f, S.psi1_f, S.psi2_f]
    jtfs_gen = joint_timefrequency_scattering(
        U_0_in,
        backend,
        filters,
        S.log2_stride,
        S.average == "local",
        S.filters_fr,
        S.log2_stride_fr,
        S.average_fr == "local",
    )

    # Zeroth order
    S_0 = next(jtfs_gen)
    loss = torch.linalg.norm(S_0["coef"])
    loss.backward(retain_graph=True)
    assert torch.abs(loss) >= 0.0
    grad = x.grad
    assert torch.max(torch.abs(grad)) > 0.0

    for S in jtfs_gen:
        loss = torch.linalg.norm(S["coef"])
        loss.backward(retain_graph=True)
        assert torch.abs(loss) >= 0.0
        grad = x.grad
        assert torch.max(torch.abs(grad)) > 0.0


def test_differentiability_jtfs_tensorflow(random_state=42):
    device = "cpu"
    J = 8
    J_fr = 3
    shape = (4096,)
    Q = 8

    with tf.device(device):
        S = TimeFrequencyScattering(
            J=J, J_fr=J_fr, shape=shape, Q=Q, T=0, F=0, frontend="tensorflow"
        )
        S.build()
        S.create_filters()
        x = tf.Variable(tf.random.normal(shape))
    backend = S.backend
    tf.random.set_seed(random_state)

    filters = [S.phi_f, S.psi1_f, S.psi2_f]

    with tf.GradientTape(persistent=True) as tape:
        x_shape = backend.shape(x)
        _, signal_shape = x_shape[:-1], x_shape[-1:]
        x_reshaped = backend.reshape_input(x, signal_shape)
        U_0_in = backend.pad(x_reshaped, pad_left=S.pad_left, pad_right=S.pad_right)

        jtfs_gen = joint_timefrequency_scattering(
            U_0_in,
            backend,
            filters,
            S.log2_stride,
            S.average == "local",
            S.filters_fr,
            S.log2_stride_fr,
            S.average_fr == "local",
        )
        # Zeroth order
        S_0 = next(jtfs_gen)
        loss = tf.norm(S_0["coef"])
        assert tf.abs(loss) >= 0.0
        losses = [loss]
        for S in list(jtfs_gen):
            loss = tf.norm(S["coef"])
            assert tf.abs(loss) >= 0.0
            losses.append(loss)
    for loss in losses:
        grad = tape.gradient(loss, x)
        assert tf.reduce_max(tf.abs(grad)) > 0.0


frontends = ["numpy", "sklearn"]
@pytest.mark.parametrize("frontend", frontends)
def test_jtfs_numpy_and_sklearn(frontend):
    # Test __init__
    kwargs = {"J": 8, "J_fr": 3, "shape": (1024,), "Q": 3}
    x = np.zeros(kwargs["shape"])
    x[kwargs["shape"][0] // 2] = 1

    # Local averaging
    S = TimeFrequencyScattering(frontend=frontend, **kwargs)
    assert S.F == (2**S.J_fr)
    Sx = S(x)
    assert isinstance(Sx, np.ndarray)
    assert Sx.ndim == 2

    # Global averaging
    S = TimeFrequencyScattering(frontend=frontend, T="global", **kwargs)
    Sx = S(x)
    assert Sx.ndim == 2
    assert Sx.shape[-1] == 1

    # Dictionary output
    S = TimeFrequencyScattering(frontend=frontend, out_type="dict", **kwargs)
    Sx = S(x)
    assert all([isinstance(path, np.ndarray) for path in Sx.values()])

    # List output
    S = TimeFrequencyScattering(frontend=frontend, out_type="list", T=0, F=0, **kwargs)
    Sx = S(x)
    assert all([path["coef"].ndim == 1 for path in Sx if path["order"] > 0])

    # meta()
    meta = S.meta()
    for key in ["xi", "sigma", "j", "xi_fr", "sigma_fr", "j_fr", "spin"]:
        assert key in meta.keys()
        assert len(meta[key]) == len(Sx)

    # format='time'
    S = TimeFrequencyScatteringNumPy(T=None, F=0, format="time", **kwargs)
    Sx = S(x)
    assert Sx.ndim == 2

    # format='joint'
    S = TimeFrequencyScatteringNumPy(T=None, format="joint", **kwargs)
    Sx = S(x)
    assert Sx.ndim == 3

    # format='joint' meta()
    meta = S.meta()
    assert len(meta["order"]) == Sx.shape[-3]

    # format='time' with global averaging
    S = TimeFrequencyScatteringNumPy(T="global", F=0, format="time", **kwargs)
    Sx = S(x)
    assert Sx.ndim == 2

    # format='time' meta()
    meta = S.meta()
    assert len(meta["key"]) == Sx.shape[-2]

    # Frontend dispatch
    S_entry = TimeFrequencyScattering(frontend=frontend, **kwargs)(x)
    S_numpy = TimeFrequencyScatteringNumPy(**kwargs)(x)
    assert np.allclose(S_entry, S_numpy)


frontends = ["torch", "tensorflow"]
@pytest.mark.parametrize("frontend", frontends)
def test_jtfs_torch_tf_frontends(frontend):
    # Test __init__
    kwargs = {"J": 8, "J_fr": 3, "shape": (8192,), "Q": 3}
    x = torch.zeros(kwargs["shape"])
    x[kwargs["shape"][0] // 2] = 1

    # format='time'
    S = TimeFrequencyScattering(frontend=frontend, T=None, F=0, format="time", **kwargs)
    Sx = S(x)
    assert Sx.ndim == 2

    # format='time' with global averaging
    S = TimeFrequencyScattering(frontend=frontend, T="global", F=0, format="time", **kwargs)
    Sx = S(x)
    assert Sx.ndim == 2

    # Local averaging
    S = TimeFrequencyScattering(frontend=frontend, format="joint", **kwargs)
    assert S.F == (2**S.J_fr)
    Sx = S(x)
    assert isinstance(Sx, torch.Tensor) if frontend == "torch" else isinstance(Sx, tf.Tensor)
    assert Sx.ndim == 3


def test_jtfs_jax_frontend():
    # Test __init__
    kwargs = {"J": 8, "J_fr": 3, "shape": (8192,), "Q": 3}
    x = np.zeros(kwargs["shape"])
    x[kwargs["shape"][0] // 2] = 1
    x = device_put(jnp.asarray(x))

    # Local averaging
    S = TimeFrequencyScatteringJax(format="joint", **kwargs)
    assert S.F == (2**S.J_fr)
    Sx = S(x)
    assert Sx.ndim == 3
    assert S.backend.name == 'jax'


frontends = ["torch", "tensorflow"]
@pytest.mark.parametrize("frontend", frontends)
def test_F(frontend):
    """
    Applies scattering on a stored signal with non-default F
    """
    kwargs = {"J": 8, "J_fr": 3, "shape": (8192,), "Q": 3}
    x = torch.zeros(kwargs["shape"])
    x[kwargs["shape"][0] // 2] = 1

    # default F
    jtfs0 = TimeFrequencyScattering(frontend=frontend, format="joint", **kwargs)

    # explicit F
    jtfs1 = TimeFrequencyScattering(frontend=frontend, format="joint", F=2**kwargs['J_fr'], **kwargs)

    Sx0 = jtfs0(x)
    Sx1 = jtfs1(x)
    assert torch.equal(Sx0, Sx1) if frontend == "torch" else tf.reduce_all(tf.equal(Sx0,Sx1))
    assert Sx0.shape == Sx1.shape

    """ non-default F smaller than 2**J_fr. This checks that the subsampling
        index does not exceed log2_F and cause an IndexError at runtime (#976).
    """
    jtfs2 = TimeFrequencyScattering(frontend="torch", format="joint", F=2**kwargs['J_fr'] // 4, **kwargs)
    Sx2 = jtfs2(x)
    assert Sx0.shape[0] == Sx2.shape[0] and Sx0.shape[2] == Sx2.shape[2] and Sx0.shape[1] < Sx2.shape[1]


backends = ["numpy", "torch", "tensorflow", "jax", "sklearn"]
@pytest.mark.parametrize("backend", backends)
def test_986(backend):
    J_fr = 6
    jtfs = TimeFrequencyScatteringBase(
        J=8, J_fr=J_fr, shape=4096, Q=8, backend=backend
    )
    jtfs.build()
    jtfs.create_filters()

    psis = jtfs.filters_fr[1]
    positive_spins = sorted([psi for psi in psis[1:] if np.sign(psi['xi']) > 0], key=lambda x: x["xi"], reverse=True)
    negative_spins = sorted([psi for psi in psis[1:] if np.sign(psi['xi']) < 0], key=lambda x: x["xi"])
    assert all(
        [
            (psi_pos["j"] == psi_neg["j"])
            for psi_pos, psi_neg in zip(positive_spins, negative_spins)
        ]
    )
