
""" This script will test the submodules used by the scattering module"""
import os
import io
import numpy as np
import jax.numpy as jnp
import pytest
from kymatio import HarmonicScattering3D


from kymatio.scattering3d.backend.jax_backend import backend


def relative_difference(a, b):
    return jnp.sum(jnp.abs(a - b)) / max(jnp.sum(jnp.abs(a)), jnp.sum(jnp.abs(b)))


def test_against_standard_computations():
    file_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(file_path, 'test_data_3d.npz'), 'rb') as f:
        buffer = io.BytesIO(f.read())
        data = np.load(buffer)
    x = jnp.asarray(data['x'])
    scattering_ref = jnp.asarray(data['Sx'])
    J = data['J']
    L = data['L']
    integral_powers = data['integral_powers']

    M = x.shape[1]

    batch_size = x.shape[0]

    N, O = M, M
    sigma = 1

    scattering = HarmonicScattering3D(J=J, shape=(M, N, O), L=L,
                                      sigma_0=sigma, method='integral',
                                      integral_powers=integral_powers,
                                      max_order=2, backend=backend,
                                      frontend='jax')

    order_0 = backend.compute_integrals(x, integral_powers)
    scattering.max_order = 2
    scattering.method = 'integral'
    scattering.integral_powers = integral_powers
    orders_1_and_2 = scattering(x)

    order_0 = order_0.reshape((batch_size, -1))
    start = 0
    end = order_0.shape[1]
    order_0_ref = scattering_ref[:, start:end]

    orders_1_and_2 = orders_1_and_2.reshape((batch_size, -1))
    start = end
    end += orders_1_and_2.shape[1]
    orders_1_and_2_ref = scattering_ref[:, start:end]

    order_0_diff_cpu = relative_difference(order_0_ref, order_0)
    orders_1_and_2_diff_cpu = relative_difference(
        orders_1_and_2_ref, orders_1_and_2)

    assert order_0_diff_cpu < 1e-6, "CPU : order 0 do not match, diff={}".format(order_0_diff_cpu)
    assert orders_1_and_2_diff_cpu < 1e-6, "CPU : orders 1 and 2 do not match, diff={}".format(orders_1_and_2_diff_cpu)
