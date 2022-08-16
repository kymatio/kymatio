import pytest
import torch
from kymatio.torch import Scattering1D

from kymatio.scattering1d.core.scattering1d import scattering1d
from kymatio.scattering1d.core.timefrequency_scattering import scattering1d_widthfirst
from kymatio.scattering1d.frontend.base_frontend import TimeFrequencyScatteringBase


def test_jtfs_build():
    # Test __init__
    jtfs = TimeFrequencyScatteringBase(
        J=10, J_fr=3, shape=4096, Q=8, backend='torch')
    assert jtfs.F is None

    # Test Q_fr
    jtfs.build()
    assert jtfs.Q_fr == (1,)

    # Test parse_T(self.F, self.J_fr, N_input_fr, T_alias='F')
    assert jtfs.F == (2 ** jtfs.J_fr)
    assert jtfs.log2_F == jtfs.J_fr

    # Test that frequential padding is at least six times larger than F
    N_input_fr = len(jtfs.psi1_f)
    assert jtfs._N_padded_fr > (N_input_fr + 6 * jtfs.F)

    # Test that padded frequency domain is divisible by max subsampling factor
    assert (jtfs._N_padded_fr % (2**jtfs.J_fr)) == 0


def test_Q():
    jtfs_kwargs = dict(J=10, J_fr=3, shape=4096, Q=8, backend='torch')
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


def test_jtfs_create_filters():
    jtfs = TimeFrequencyScatteringBase(
        J=10, J_fr=3, shape=4096, Q=8, backend='torch')
    jtfs.build()
    jtfs.create_filters()

    phi = jtfs.filters_fr[0]
    assert phi['N'] == jtfs._N_padded_fr


def test_scattering1d_widthfirst():
    """Checks that width-first and depth-first algorithms have same output."""
    J = 5
    shape = (1024,)
    S = Scattering1D(J, shape)
    x = torch.zeros(shape)
    x[shape[0]//2] = 1

    ### average_local == False ###
    average_local = False
    # Width-first scattering
    x_shape = S.backend.shape(x)
    batch_shape, signal_shape = x_shape[:-1], x_shape[-1:]
    x = S.backend.reshape_input(x, signal_shape)
    U_0 = S.backend.pad(x, pad_left=S.pad_left, pad_right=S.pad_right)
    filters = [S.phi_f, S.psi1_f, S.psi2_f]
    U_gen = scattering1d_widthfirst(U_0, S.backend, filters, S.oversampling,
        average_local)
    U_width = {path['n']: path['coef'] for path in U_gen}

    # Depth-first scattering
    U_gen = scattering1d(U_0, S.backend, filters, S.oversampling, average_local)
    U_depth = {path['n']: path['coef'] for path in U_gen}

    # Check orders 0 and 1
    skip_order2 = lambda item: (len(item[0]) < 2)
    U_width_no_order2 = dict(filter(skip_order2, U_width.items()))
    U_depth_no_order2 = dict(filter(skip_order2, U_depth.items()))
    assert set(U_width_no_order2.keys()) == set(U_depth_no_order2.keys())
    for key in U_width_no_order2:
        assert torch.allclose(U_width_no_order2[key], U_depth_no_order2[key])

    # Check order 2
    keep_order2 = lambda item: (len(item[0]) == 2)
    Y_width_order2 = dict(filter(keep_order2, U_width.items()))
    U_depth_order2 = dict(filter(keep_order2, U_depth.items()))
    for key in Y_width_order2:
        n2 = key[-1]
        keep_n2 = lambda item: (item[0][-1] == n2)
        U_n2 = dict(filter(keep_n2, U_depth_order2.items()))
        U_n2 = S.backend.concatenate([U_n2[key] for key in sorted(U_n2.keys())])
        assert torch.allclose(S.backend.modulus(Y_width_order2[key]), U_n2)

    ### average_local == True ###
    average_local = True

    # Width-first
    S_gen = scattering1d_widthfirst(U_0, S.backend, filters, S.oversampling,
        average_local)
    S_width = {path['n']: path['coef'] for path in S_gen}

    # Depth-first
    S_gen = scattering1d(U_0, S.backend, filters, S.oversampling, average_local)
    S1_depth = {path['n']: path['coef'] for path in S_gen if len(path['n']) > 0}

    # Check order 1 (order 2 is unaveraged in width-first so irrelevant here)
    keep_order1 = lambda item: (len(item[0]) == 1)
    S1_width = dict(filter(keep_order1, S_width.items()))
    assert len(S1_width) == 1
    S1_width = S1_width[(-1,)]
    S1_depth = S.backend.concatenate([
        S1_depth[key] for key in sorted(S1_depth.keys()) if len(key)==1])
    assert torch.allclose(S1_width, S1_depth)
