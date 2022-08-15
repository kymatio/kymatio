import pytest
import torch
from kymatio.torch import Scattering1D

from kymatio.scattering1d.core.scattering1d import scattering1d
from kymatio.scattering1d.core.timefrequency_scattering import scattering1d_widthfirst

def test_scattering1d_widthfirst():
    """Checks that width-first and depth-first algorithms have same output."""
    J = 5
    shape = (1024,)
    S = Scattering1D(J, shape)
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

    # Check order 2
    keep_order2 = lambda item: (len(item[0]) == 2)
    Y_width_order2 = dict(filter(keep_order2, W.items()))
    for key in Y_width_order2:
        n2 = key[-1]
        keep_n2 = lambda item: (item[0][-1] == n2)
        U_n2 = dict(filter(keep_n2, U2_depth.items()))
        U_n2 = S.backend.concatenate([U_n2[key] for key in sorted(U_n2.keys())])
        assert torch.allclose(S.backend.modulus(Y_width_order2[key]), U_n2)

    # Check order 0 under average_local=False
    W_gen = scattering1d_widthfirst(U_0, S.backend, filters, S.oversampling,
        average_local=False)
    U_0_width = next(W_gen)
    assert torch.allclose(U_0_width['coef'], U_0)
