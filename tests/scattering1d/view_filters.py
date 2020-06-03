from kymatio import Scattering1D
import os
import numpy as np
import io
import jax.numpy as jnp
from jax import device_put
import matplotlib.pyplot as plt

test_data_dir = os.path.dirname(__file__)
with open(os.path.join(test_data_dir, 'test_data_1d.npz'), 'rb') as f:
    buffer = io.BytesIO(f.read())
    data = np.load(buffer)

x = device_put(jnp.asarray(data['x']))
J = data['J']
Q = data['Q']
Sx0 = device_put(jnp.asarray(data['Sx']))
T = x.shape[-1]
scattering = Scattering1D(J, T, Q, backend='jax', frontend='jax')
Sx = scattering(x)
S_diff = Sx - Sx0

#assert jnp.allclose(Sx, Sx0)
norms = np.linalg.norm(S_diff.reshape(S_diff.shape[:-1] + (-1,)), axis=-1, ord=np.inf)
S_diff = S_diff/(1e-19+norms[..., np.newaxis])
meta = scattering.meta()
order0 = np.where(meta['order'] == 0)
order1 = np.where(meta['order'] == 1)
order2 = np.where(meta['order'] == 2)

plt.figure()
plt.subplot(3,1,1)
plt.plot(S_diff[order0][0])
plt.title('Zeroth-order scattering')
plt.subplot(3,1,2)
plt.imshow(S_diff[order1][0], aspect='auto')
plt.title('First-order scattering')
plt.subplot(3,1,3)
plt.imshow(S_diff[order2][0], aspect='auto')
plt.title('Second-order scattering')

plt.savefig("panel.png")
