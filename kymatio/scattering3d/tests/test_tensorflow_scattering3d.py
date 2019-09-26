import os
import io
import tensorflow as tf
from kymatio import HarmonicScattering3D
import torch
import numpy as np
import pytest

def relative_difference(a, b):
    return np.sum(np.abs(a - b)) / max(np.sum(np.abs(a)), np.sum(np.abs(b)))

def test_against_standard_computations():
    file_path = os.path.abspath(os.path.dirname(__file__))
    data = torch.load(os.path.join(file_path, 'test_data_3d.pt'))
    x = data['x'].numpy()
    scattering_ref = data['Sx'].numpy()
    J = data['J']
    L = data['L']
    integral_powers = data['integral_powers']

    M = x.shape[1]

    batch_size = x.shape[0]
    x_tf = tf.compat.v1.placeholder(tf.complex64, shape=x.shape)


    N, O = M, M
    sigma = 1
    from kymatio.scattering3d.backend import tensorflow_backend

    scattering = HarmonicScattering3D(J=J, shape=(M, N, O), L=L, sigma_0=sigma, method='integral', integral_powers=integral_powers, max_order=2, frontend='tensorflow')
    S_tf  = scattering(x_tf)
    
    # Create session
    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    orders_1_and_2  = sess.run(S_tf, feed_dict={x_tf: x})
    sess.close()
    
    #order_0 = numpy_backend.compute_integrals(x, integral_powers)   
    # Extract orders and make order axis the slowest in accordance with
    # the stored reference scattering transform.

    # WARNING: These are hard-coded values for the setting J = 2.
    n_order_1 = 3
    n_order_2 = 3

    order_1 = orders_1_and_2[:,0:n_order_1,...]
    order_2 = orders_1_and_2[:,n_order_1:n_order_1+n_order_2,...]

    #TODO: order_0 test
    start = 0
    end = 2
    order_0_ref = scattering_ref[:,start: end]
    # Permute the axes since reference has (batch index, integral power, j,
    # ell) while the computed transform has (batch index, j, ell, integral
    # power).
    order_1 = order_1.transpose((0, 3, 1, 2))
    order_2 = order_2.transpose((0, 3, 1, 2))
    
    order_1 = order_1.reshape((batch_size, -1))
    order_2 = order_2.reshape((batch_size, -1))
    
    orders_1_and_2 = np.concatenate((order_1, order_2), 1)
    orders_1_and_2 = orders_1_and_2.reshape((batch_size, -1))

    start = end
    end += orders_1_and_2.shape[1]
    orders_1_and_2_ref = scattering_ref[:,start: end]

    orders_1_and_2_diff_cpu = relative_difference(orders_1_and_2_ref, orders_1_and_2)
    assert orders_1_and_2_diff_cpu < 1e-6, "Tensorflow : orders 1 and 2 do not match, diff={}".format(orders_1_and_2_diff_cpu)
