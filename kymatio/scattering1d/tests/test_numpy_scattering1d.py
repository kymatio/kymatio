import os
import numpy as np
from kymatio.scattering1d import Scattering1D
import torch

class TestScattering1DNumpy:
    def test_Scattering1D(self):
        test_data_dir = os.path.dirname(__file__)
        test_data_filename = os.path.join(test_data_dir, 'test_data_1d.pt')
        data = torch.load(test_data_filename, map_location='cpu')

        x = data['x'].numpy()
        J = data['J']
        Q = data['Q']
        Sx0 = data['Sx'].numpy()

        T = x.shape[2]

        # Convert from old (B, 1, T) format.
        x = x.squeeze(1)

        scattering = Scattering1D(J, T, Q, frontend='numpy')

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

        perm = np.concatenate(orders)

        Sx0 = Sx0[:, perm, :]

        Sx = scattering(x)

        assert np.allclose(Sx0, Sx)
