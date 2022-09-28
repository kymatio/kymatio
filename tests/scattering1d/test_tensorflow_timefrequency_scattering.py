import numpy as np
import pytest
import tensorflow as tf
import kymatio
from kymatio.scattering1d.core.scattering1d import scattering1d
from kymatio.scattering1d.filter_bank import gauss_1d
from kymatio.scattering1d.core.timefrequency_scattering import (
    joint_timefrequency_scattering,
    time_scattering_widthfirst,
)
from kymatio.scattering1d.backend.tensorflow_backend import TensorFlowBackend1D
from kymatio.scattering1d.frontend.base_frontend import TimeFrequencyScatteringBase
from kymatio.scattering1d.frontend.tensorflow_frontend import (
    TimeFrequencyScatteringTensorFlow,
)
from test_timefrequency_scattering import _joint_timefrequency_scattering_test_routine


def test_jtfs_build():
    # Test __init__
    jtfs = TimeFrequencyScatteringBase(
        J=10, J_fr=3, shape=4096, Q=8, backend="tensorflow"
    )
    assert jtfs.F is None

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


def test_Q():
    jtfs_kwargs = dict(J=10, J_fr=3, shape=4096, Q=8, backend="tensorflow")
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


def test_jtfs():
    # Define scattering object
    J = 8
    J_fr = 3
    shape = (4096,)
    Q = 8
    S = TimeFrequencyScatteringBase(
        J=J, J_fr=J_fr, shape=shape, Q=Q, T=0, F=0, backend="tensorflow"
    )
    S.build()
    S.create_filters()
    assert not S.average
    assert not S.average_fr
    backend = TensorFlowBackend1D


    _joint_timefrequency_scattering_test_routine(S, backend, shape)


def test_jtfs_create_filters():
    J_fr = 3
    jtfs = TimeFrequencyScatteringBase(
        J=10, J_fr=J_fr, shape=4096, Q=8, backend="tensorflow"
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


def test_time_scattering_widthfirst():
    """Checks that width-first and depth-first algorithms have same output."""
    J = 5
    shape = (1024,)
    S = kymatio.Scattering1D(J, shape, frontend="tensorflow")
    x = np.zeros(shape)
    x[shape[0] // 2] = 1
    x = tf.convert_to_tensor(x)
    x_shape = S.backend.shape(x)
    _, signal_shape = x_shape[:-1], x_shape[-1:]
    x = S.backend.reshape_input(x, signal_shape)
    U_0 = S.backend.pad(x, pad_left=S.pad_left, pad_right=S.pad_right)

    # Width-first
    filters = [S.phi_f, S.psi1_f, S.psi2_f]
    W_gen = time_scattering_widthfirst(
        U_0, S.backend, filters, S.oversampling, average_local=True
    )
    W = {path["n"]: path["coef"] for path in W_gen}

    # Depth-first
    S_gen = scattering1d(U_0, S.backend, filters, S.oversampling, average_local=True)
    S1_depth = {path["n"]: path["coef"] for path in S_gen if len(path["n"]) > 0}

    # Check order 1
    keep_order1 = lambda item: (len(item[0]) == 1)
    S1_width = dict(filter(keep_order1, W.items()))
    assert len(S1_width) == 1
    S1_width = S1_width[(-1,)]
    S1_depth = S.backend.concatenate(
        [S1_depth[key] for key in sorted(S1_depth.keys()) if len(key) == 1]
    )
    assert np.allclose(S1_width.numpy(), S1_depth.numpy())


def test_differentiability_jtfs(random_state=42):
    device = "cpu"
    J = 8
    J_fr = 3
    shape = (4096,)
    Q = 8

    with tf.device(device):
        S = TimeFrequencyScatteringTensorFlow(
            J=J, J_fr=J_fr, shape=shape, Q=Q, T=0, F=0
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
            S.oversampling,
            S.average == "local",
            S.filters_fr,
            S.oversampling_fr,
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
