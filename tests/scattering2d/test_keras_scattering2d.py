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
