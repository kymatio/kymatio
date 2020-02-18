import os
import io
import numpy as np

from kymatio.sklearn import Scattering2D as ScatteringTransformer2D
from kymatio.numpy import Scattering2D as ScatteringNumPy2D

def test_sklearn_transformer():
    test_data_dir = os.path.join(os.path.dirname(__file__))

    with open(os.path.join(test_data_dir, 'test_data_2d.npz'), 'rb') as f:
        buf = io.BytesIO(f.read())
        data = np.load(buf)

    x = data['x']
    J = data['J']

    S = ScatteringNumPy2D(J, x.shape[2:])
    Sx = S.scattering(x)

    x_raveled = x.reshape(x.shape[0], -1)
    Sx_raveled = Sx.reshape(x.shape[0], -1)

    st = ScatteringTransformer2D(J, x.shape[2:])

    t = st.transform(x_raveled)

    assert np.allclose(Sx_raveled, t)
