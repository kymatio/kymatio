# This script generates the file generate_test_data_1d.npz, which serves to
# ensure that different backends yield consistent results. Last breaking change: v0.4

from kymatio.numpy import Scattering1D
import numpy as np
import os

# Create signal
shape = (4, 512)
x = np.random.randn(*shape).astype('float32')

# Compute scattering transform
J = 6
Q = 16
S = Scattering1D(J=J, Q=Q, shape=shape[-1])
Sx = S(x).astype('float32')

# Export to NPZ format
data = dict(x=x, J=J, Q=Q, Sx=Sx)
test_data_dir = os.path.dirname(__file__)
test_data_path = os.path.join(test_data_dir, 'test_data_1d.npz')
np.savez(test_data_path, **data)