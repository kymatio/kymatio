import numpy as np
import pytest
import torch

import kymatio
from kymatio.scattering1d.core.scattering1d import scattering1d
from kymatio.scattering1d.core.timefrequency_scattering import (frequency_scattering,
    scattering1d_widthfirst)
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
    J_fr = 3
    jtfs = TimeFrequencyScatteringBase(
        J=10, J_fr=J_fr, shape=4096, Q=8, backend='torch')
    jtfs.build()
    jtfs.create_filters()

    phi = jtfs.filters_fr[0]
    assert phi['N'] == jtfs._N_padded_fr

    psis = jtfs.filters_fr[1]
    assert len(psis) == (1 + 2*(J_fr+1))
    assert np.isclose(psis[0]['xi'], 0)
    positive_spins = psis[1:][:J_fr+1]
    negative_spins = psis[1:][J_fr+1:]
    assert all([psi['xi']>0 for psi in positive_spins])
    assert all([psi['xi']<0 for psi in negative_spins])
    assert all([(psi_pos['xi']==-psi_neg['xi'])
        for psi_pos, psi_neg in zip(positive_spins, negative_spins)])


def test_scattering1d_widthfirst():
    """Checks that width-first and depth-first algorithms have same output."""
    J = 5
    shape = (1024,)
    S = kymatio.Scattering1D(J, shape, frontend='torch')
    x = torch.zeros(shape)
    x[shape[0]//2] = 1
    x_shape = S.backend.shape(x)
    batch_shape, signal_shape = x_shape[:-1], x_shape[-1:]
    x = S.backend.reshape_input(x, signal_shape)
    U_0 = S.backend.pad(x, pad_left=S.pad_left, pad_right=S.pad_right)

    # Width-first
    filters = [S.phi_f, S.psi1_f, S.psi2_f]
    W_gen = scattering1d_widthfirst(U_0, S.backend, filters, S.oversampling,
        average_local=True)
    W = {path['n']: path['coef'] for path in W_gen}

    # Depth-first
    S_gen = scattering1d(U_0, S.backend, filters, S.oversampling,
        average_local=True)
    S1_depth = {path['n']: path['coef'] for path in S_gen if len(path['n']) > 0}
    U_gen = scattering1d(U_0, S.backend, filters, S.oversampling,
        average_local=False)
    U2_depth = {path['n']: path['coef'] for path in U_gen if len(path['n']) == 2}

    # Check order 1
    keep_order1 = lambda item: (len(item[0]) == 1)
    S1_width = dict(filter(keep_order1, W.items()))
    assert len(S1_width) == 1
    S1_width = S1_width[(-1,)]
    S1_depth = S.backend.concatenate([
        S1_depth[key] for key in sorted(S1_depth.keys()) if len(key)==1])
    assert torch.allclose(S1_width, S1_depth)


def test_frequency_scattering():
    # Create S1 array as input
    J = 5
    shape = (4096,)
    Q = 8
    S = kymatio.Scattering1D(J=J, Q=Q, shape=shape, backend='numpy')
    x = torch.zeros(shape)
    x[shape[0]//2] = 1
    x_shape = S.backend.shape(x)
    batch_shape, signal_shape = x_shape[:-1], x_shape[-1:]
    x = S.backend.reshape_input(x, signal_shape)
    U_0 = S.backend.pad(x, pad_left=S.pad_left, pad_right=S.pad_right)
    filters = [S.phi_f, S.psi1_f, S.psi2_f]
    average_local = True
    S_gen = scattering1d(U_0, S.backend, filters, S.oversampling, average_local)
    S_1_dict = {path['n']: path['coef'] for path in S_gen if len(path['n'])==1}
    S_1 = S.backend.concatenate([S_1_dict[key] for key in sorted(S_1_dict.keys())])
    X = {'coef': S_1, 'n1_max': len(S_1_dict)}

    # Define scattering object
    J_fr = 3
    jtfs = TimeFrequencyScatteringBase(
        J=J, J_fr=J_fr, shape=shape, Q=Q, backend='numpy')
    jtfs.build()
    jtfs.create_filters()

    # spinned=False
    freq_gen = frequency_scattering(X, S.backend, jtfs.filters_fr,
        jtfs.oversampling_fr, jtfs.average_fr=='local', spinned=False)
    for Y_fr in freq_gen:
        assert Y_fr['spin'] >= 0

    # spinned=True
    X['coef'] = X['coef'].astype('complex64')
    freq_gen = frequency_scattering(X, S.backend, jtfs.filters_fr,
        jtfs.oversampling_fr, jtfs.average_fr=='local', spinned=True)
    for Y_fr in freq_gen:
        assert Y_fr['coef'].shape[-1] == X['coef'].shape[-1]
