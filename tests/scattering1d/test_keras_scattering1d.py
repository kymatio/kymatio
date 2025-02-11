import pytest
from tensorflow.keras.layers import Input, Flatten, Dense
from kymatio.keras import Scattering1D, TimeFrequencyScattering
from tensorflow.keras.models import Model
import os
import numpy as np
import io
import sys
import tensorflow as tf

def test_Scattering1D():
    """
    Applies scattering on a stored signal to make sure its output agrees with
    a previously calculated version.
    """
    test_data_dir = os.path.dirname(__file__)
    with open(os.path.join(test_data_dir, 'test_data_1d.npz'), 'rb') as f:
        buffer = io.BytesIO(f.read())
        data = np.load(buffer)
    x = data['x']
    J = data['J']
    Q = int(data['Q'])
    Sx0 = data['Sx']
    # default
    inputs0 = Input(shape=(x.shape[-1], ))
    sc0 = Scattering1D(J=J, Q=Q)(inputs0)
    model0 = Model(inputs0, sc0)
    model0.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    Sg0 = model0.predict(x)
    assert np.allclose(Sg0, Sx0, atol=1e-06)
    # adjust T
    sigma_low_scale_factor = 2
    T = 2**(J-sigma_low_scale_factor)
    inputs1 = Input(shape=(x.shape[-1], ))
    sc1 = Scattering1D(J=J, Q=Q, T=T)(inputs1)
    model1 = Model(inputs1, sc1)
    model1.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    Sg1 = model1.predict(x)
    assert Sg1.shape == (
        Sg0.shape[0], Sg0.shape[1], Sg0.shape[2]*2**(sigma_low_scale_factor))

    save_stdout = sys.stdout
    result = io.StringIO()
    sys.stdout = result
    model1.summary()
    sys.stdout = save_stdout
    assert 'scattering1d' in result.getvalue()
  
    sc0 = Scattering1D(J=J, Q=Q)
    sc0.build(inputs0.shape)
    assert sc0.compute_output_shape(inputs0.shape)[-1] == 8

def test_Q():
    J = 3
    length = 1024
    inputs = Input(shape=(length,))

    # test different cases for Q
    with pytest.raises(ValueError) as ve:
        S = Scattering1D(J=J, Q=0.9)(inputs)
        Q = S.Q
    assert "Q must always be >= 1" in ve.value.args[0]

    with pytest.raises(ValueError) as ve:
        S = Scattering1D(J=J, Q=[8])(inputs)
        Q = S.Q
    assert "Q must be an integer or a tuple" in ve.value.args[0]

    Sc_int = Scattering1D(J=J, Q=(8, ))(inputs)
    Sc_tuple = Scattering1D(J=J, Q=(8, 1))(inputs)

    assert Sc_int.shape[1] == Sc_tuple.shape[1]

    # test dummy input
    x = np.zeros((1, length))
    model0 = Model(inputs, Sc_int)
    model0.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    Sc_int_out = model0.predict(x)

    model1 = Model(inputs, Sc_tuple)
    model1.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    Sc_tuple_out = model1.predict(x)

    assert Sc_int_out.shape == (Sc_tuple_out.shape[0], Sc_tuple_out.shape[1], Sc_tuple_out.shape[2])


def test_TimeFrequencyScattering():
    """
    Applies scattering on a stored signal to make sure its output agrees with
    a previously calculated version.
    """
    test_data_dir = os.path.dirname(__file__)
    with open(os.path.join(test_data_dir, 'test_data_1d.npz'), 'rb') as f:
        buffer = io.BytesIO(f.read())
        data = np.load(buffer)
    x = data['x']
    J = data['J']
    Q = int(data['Q'])
    Sx0 = data['Sx']
    # default
    inputs0 = Input(shape=(x.shape[-1], ))
    sc0 = TimeFrequencyScattering(J=J, J_fr=1, Q=Q)(inputs0)
    model0 = Model(inputs0, sc0)
    model0.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    Sg0 = model0.predict(x)
    assert np.allclose(Sg0, Sx0, atol=1e-06)
    # adjust T
    sigma_low_scale_factor = 2
    T = 2**(J-sigma_low_scale_factor)
    inputs1 = Input(shape=(x.shape[-1], ))
    sc1 = TimeFrequencyScattering(J=J, J_fr=0, Q=Q, T=T)(inputs1)
    model1 = Model(inputs1, sc1)
    model1.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    Sg1 = model1.predict(x)
    assert Sg1.shape == (
        Sg0.shape[0], Sg0.shape[1], Sg0.shape[2]*2**(sigma_low_scale_factor))

    save_stdout = sys.stdout
    result = io.StringIO()
    sys.stdout = result
    model1.summary()
    sys.stdout = save_stdout
    assert 'timefrequencyscattering' in result.getvalue()
  
    sc0 = TimeFrequencyScattering(J=J, J_fr=0, Q=Q)
    sc0.build(inputs0.shape)
    assert sc0.compute_output_shape(inputs0.shape)[-1] == 8


from kymatio.scattering1d.frontend.tensorflow_frontend import TimeFrequencyScatteringTensorFlow
from kymatio.scattering1d.frontend.tensorflow_frontend import ScatteringTensorFlow1D
import torch

def test_jtfs_torch_tf_frontends():
    # Test __init__
    kwargs = {"J": 8, "J_fr": 3, "Q": 3}
    shape = (8192,)
    x = np.zeros((1, 8192,))
    x[:, shape[0] // 2] = 1
    print(type(x))
    tf.compat.v1.enable_eager_execution()  # Fixes eager execution
    x = tf.convert_to_tensor(x)
    inputs0 = Input(shape=(x.shape[-1], ))
    # format='time'
    S = TimeFrequencyScattering(T=None, F=0, format="time", **kwargs)(inputs0)
    model0 = Model(inputs0, S)
    model0.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    Sx = model0.predict(x)
    print(Sx.shape)
    print(type(Sx))
    assert Sx.ndim == 3

    # format='time' with global averaging
    inputs1 = Input(shape=(x.shape[-1], ))
    S = TimeFrequencyScattering(T="global", F=0, format="time", **kwargs)(inputs1)
    model1 = Model(inputs1, S)
    model1.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    Sx = model1.predict(x)
    print(Sx.shape)
    print(type(Sx))
    assert Sx.ndim == 3

    # Local averaging
    inputs2 = Input(shape=(x.shape[-1], ))
    S = TimeFrequencyScattering(format="joint", **kwargs)(inputs2)
    model2 = Model(inputs2, S)
    model2.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    Sx = model1.predict(x)

    assert S.F == (2**S.J_fr)
    print(Sx.shape)
    print(type(Sx))
    Sx = S(x)
    assert isinstance(Sx, torch.Tensor) if frontend == "torch" else isinstance(Sx, tf.Tensor)
    assert Sx.ndim == 4


