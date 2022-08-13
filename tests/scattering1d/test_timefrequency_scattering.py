import pytest
import torch
from kymatio.torch import Scattering1D

from kymatio.scattering1d.core.scattering1d import scattering1d
from kymatio.scattering1d.core.timefrequency_scattering import scattering1d_widthfirst

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
def test_scattering1d_widthfirst(device, backend):
    """Checks that width-first and depth-first algorithms have same output."""
    J = 5
    shape = (1024,)
    S = Scattering1D(J, shape, backend=backend).to(device)
    x = torch.zeros(shape)
    x[shape[0]//2] = 1

    # Width-first scattering
    x_shape = S.backend.shape(x)
    batch_shape, signal_shape = x_shape[:-1], x_shape[-1:]
    x = S.backend.reshape_input(x, signal_shape)
    U_0 = S.backend.pad(x, pad_left=S.pad_left, pad_right=S.pad_right)
    filters = [S.phi_f, S.psi1_f, S.psi2_f]
    U_gen = scattering1d_widthfirst(U_0, S.backend, filters, S.oversampling,
        average_local=False)
    U_width = {path['n']: path['coef'] for path in U_gen}

    # Depth-first scattering
    U_gen = scattering1d(U_0, S.backend, S.psi1_f, S.psi2_f, S.phi_f,
        S.oversampling, S.max_order, average=False)
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
