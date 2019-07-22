import os
import io
import numpy as np
from kymatio.scattering2d import Scattering2D_numpy
import pickle
import torch

def reorder_coefficients_from_interleaved(J, L):
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
def test_Scattering2D():
    test_data_dir = os.path.dirname(__file__)
    data = None
    with open(os.path.join(test_data_dir, 'test_data_2d.pt'), 'rb') as f:
        buffer = io.BytesIO(f.read())
        data = torch.load(buffer)

    x = data['x']
    S = data['Sx']
    J = data['J']

    # we need to reorder S from interleaved (how it's saved) to o0, o1, o2
    # (which is how it's now computed)

    o0, o1, o2 = reorder_coefficients_from_interleaved(J, L=8)
    reorder = np.concatenate((o0, o1, o2))
    S = S[..., reorder, :, :]

    pre_pad = data['pre_pad']

    M = x.shape[2]
    N = x.shape[3]

    # Then, let's check when using pure pytorch code
    scattering = Scattering2D_numpy(J, shape=(M, N), pre_pad=pre_pad)

    x = x
    S = S
    Sg = scattering(x)
    assert (Sg - S).abs().max() < 1e-6