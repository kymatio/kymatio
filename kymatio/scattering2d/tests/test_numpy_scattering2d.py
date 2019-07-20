import os
import numpy as np
import pytest

try:
    from kymatio.scattering2d.backend import skcuda_b
    backends.append(skcuda_b)
except:
    pass

try:
    from kymatio.scattering2d.backend import torch_backend
    backends.append(torch_backend)
except:
    pass



# Check the scattering
# FYI: access the two different tests in here by setting envs
# KYMATIO_BACKEND=skcuda and KYMATIO_BACKEND=torch
def test_Scattering2D():
    test_data_dir = os.path.dirname(__file__)
    data = torch.load(os.path.join(test_data_dir, 'test_data_2d.pt'))

    x = data['x']
    S = data['Sx']
    J = data['J']

    # we need to reorder S from interleaved (how it's saved) to o0, o1, o2
    # (which is how it's now computed)

    o0, o1, o2 = reorder_coefficients_from_interleaved(J, L=8)
    reorder = torch.from_numpy(np.concatenate((o0, o1, o2)))
    S = S[..., reorder, :, :]

    pre_pad = data['pre_pad']

    M = x.shape[2]
    N = x.shape[3]

    import kymatio.scattering2d.backend as backend

    if backend.BACKEND_NAME == 'skcuda':
        print('skcuda backend tested!')
        # First, let's check the Jit
        scattering = Scattering2D(J, shape=(M, N), pre_pad=pre_pad)
        scattering.cuda()
        x = x.cuda()
        S = S.cuda()
        y = scattering(x)
        assert ((S - y)).abs().max() < 1e-6
    elif backend.BACKEND_NAME == 'torch':
        # Then, let's check when using pure pytorch code
        scattering = Scattering2D(J, shape=(M, N), pre_pad=pre_pad)
        Sg = []

        for device in devices:
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
            assert (Sg - S).abs().max() < 1e-6