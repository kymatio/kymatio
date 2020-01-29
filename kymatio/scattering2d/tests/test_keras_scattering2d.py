import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from kymatio.keras import Scattering2D
import os, io
import numpy as np

def test_Scattering2D():
    test_data_dir = os.path.dirname(__file__)
    data = None
    with open(os.path.join(test_data_dir, 'test_data_2d.npz'), 'rb') as f:
        buffer = io.BytesIO(f.read())
        data = np.load(buffer)

    x = data['x']
    S = data['Sx']
    J = data['J']

    # we need to reorder S from interleaved (how it's saved) to o0, o1, o2
    # (which is how it's now computed)

    o0, o1, o2 = reorder_coefficients_from_interleaved(J, 8)
    reorder = np.concatenate((o0, o1, o2))
    S = S[..., reorder, :, :]

    pre_pad = data['pre_pad']

    M = x.shape[2]
    N = x.shape[3]

    inputs = Input(shape=(3, M, N))
    scat = Scattering2D(J=J)(inputs)

    model = Model(inputs, scat)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    x = x
    S = S
    Sg = model.predict(x)
    assert np.allclose(Sg, S)


def reorder_coefficients_from_interleaved( J, L):
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
