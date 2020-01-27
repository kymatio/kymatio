import os
import torch

from kymatio.utils import ScatteringTransformer
from kymatio.scattering2d import Scattering2D
import io
import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.utils.estimator_checks import check_estimator

def test_sklearn_transformer():
    test_data_dir = os.path.join(os.path.dirname(__file__), 
                                    "..", "scattering2d",
                                    "tests")
    with open(os.path.join(test_data_dir, 'test_data_2d.npz'), 'rb') as f:
        buffer = io.BytesIO(f.read())
        data = np.load(buffer)

    x = torch.from_numpy(data['x'])
    J = data['J']

    S = Scattering2D(J, x.shape[2:], frontend='torch')
    Sx = S.forward(x)

    x_raveled = x.reshape(x.shape[0], -1).detach().cpu().numpy()
    Sx_raveled = Sx.reshape(x.shape[0], -1).detach().cpu().numpy()

    st = ScatteringTransformer(S, x[0].shape,'torch').fit()

    t = st.transform(x_raveled)
    assert_array_almost_equal(Sx_raveled, t)

    # Check numpy
    S = Scattering2D(J, x.shape[2:], frontend='numpy')
    st = ScatteringTransformer(S, x[0].shape, 'numpy').fit()

    t = st.transform(x_raveled)

    assert_array_almost_equal(Sx_raveled, t)
