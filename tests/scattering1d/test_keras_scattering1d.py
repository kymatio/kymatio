import pytest
from tensorflow.keras.layers import Input, Flatten, Dense
from kymatio.keras import Scattering1D, TimeFrequencyScattering
from tensorflow.keras.models import Model
import os
import numpy as np
import io
import sys

from kymatio.scattering1d.frontend.tensorflow_frontend import (
    TimeFrequencyScatteringTensorFlow)
from kymatio.scattering1d.frontend.numpy_frontend import (
    TimeFrequencyScatteringNumPy)

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

    assert Sc_int_out.shape == (Sc_int_out.shape[0], Sc_tuple_out.shape[1], Sc_tuple_out.shape[2])


def _jtfs_model(x, **kwargs):
    """Build and compile a Keras model wrapping a TimeFrequencyScattering
    layer, returning the model and its prediction on ``x``."""
    inputs = Input(shape=(x.shape[-1], ))
    out = TimeFrequencyScattering(**kwargs)(inputs)
    model = Model(inputs, out)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model, model.predict(x)


def _assert_output_shape_matches(x, **kwargs):
    """compute_output_shape (static inference) must agree with the realized
    transform on every non-batch axis."""
    _, Sx = _jtfs_model(x, **kwargs)
    layer = TimeFrequencyScattering(**kwargs)
    layer.build((None, x.shape[-1]))
    inferred = layer.compute_output_shape((None, x.shape[-1])).as_list()
    assert tuple(inferred[1:]) == tuple(Sx.shape[1:])
    return Sx


def test_TimeFrequencyScattering():
    """format='time': output is (batch, n_coefficients, time) and
    compute_output_shape tracks the realized shape under default, explicit and
    global temporal averaging."""
    test_data_dir = os.path.dirname(__file__)
    with open(os.path.join(test_data_dir, 'test_data_1d.npz'), 'rb') as f:
        buffer = io.BytesIO(f.read())
        data = np.load(buffer)
    x = data['x']
    J = int(data['J'])
    Q = int(data['Q'])

    # default temporal averaging (T = 2 ** J)
    model0, Sg0 = _jtfs_model(x, J=J, J_fr=1, Q=Q)
    assert Sg0.ndim == 3
    Sg0_check = _assert_output_shape_matches(x, J=J, J_fr=1, Q=Q)
    assert Sg0_check.shape == Sg0.shape

    # explicit T < 2 ** J reduces temporal subsampling, lengthening the time
    # axis by 2 ** sigma_low_scale_factor while keeping the coefficient count.
    sigma_low_scale_factor = 2
    T = 2 ** (J - sigma_low_scale_factor)
    Sg1 = _assert_output_shape_matches(x, J=J, J_fr=1, Q=Q, T=T)
    assert Sg1.shape[:2] == Sg0.shape[:2]
    assert Sg1.shape[-1] == Sg0.shape[-1] * 2 ** sigma_low_scale_factor

    # global averaging collapses the time axis to a single sample
    Sg2 = _assert_output_shape_matches(x, J=J, J_fr=1, Q=Q, T='global')
    assert Sg2.shape[-1] == 1

    # the layer is named after the transform in the model summary
    save_stdout = sys.stdout
    result = io.StringIO()
    sys.stdout = result
    model0.summary()
    sys.stdout = save_stdout
    assert 'scattering' in result.getvalue().lower()


def test_TimeFrequencyScattering_time_largeJ():
    """format='time' with a larger transform and no frequency averaging
    (F=0), including global averaging."""
    J, J_fr, Q, N = 8, 3, 3, 8192
    x = np.zeros((2, N), dtype='float32')
    x[:, N // 2] = 1.0

    Sx = _assert_output_shape_matches(x, J=J, J_fr=J_fr, Q=Q, F=0, format='time')
    assert Sx.ndim == 3

    Sx_global = _assert_output_shape_matches(
        x, J=J, J_fr=J_fr, Q=Q, T='global', F=0, format='time')
    assert Sx_global.ndim == 3
    assert Sx_global.shape[-1] == 1


def test_TimeFrequencyScattering_joint():
    """format='joint': output is (batch, n_jtfs, n_freq, time) and
    compute_output_shape matches the realized 4D shape."""
    J, J_fr, Q, N = 8, 3, 3, 8192
    x = np.zeros((2, N), dtype='float32')
    x[:, N // 2] = 1.0

    Sx = _assert_output_shape_matches(x, J=J, J_fr=J_fr, Q=Q, format='joint')
    assert Sx.ndim == 4


def test_TimeFrequencyScattering_get_config():
    """get_config serializes only the static (filterbank-determining)
    parameters and round-trips through from_config."""
    sc = TimeFrequencyScattering(J=8, J_fr=3, Q=3, Q_fr=1, format='joint')
    config = sc.get_config()
    assert set(config) == {'J', 'J_fr', 'Q', 'Q_fr', 'format'}
    assert config['Q'] == 3 and config['format'] == 'joint'

    sc_restored = TimeFrequencyScattering.from_config(config)
    assert sc_restored.get_config() == config


def _jtfs_frontend_outputs(x, **kwargs):
    """Compute the JTFS of ``x`` with the Keras, TensorFlow and NumPy frontends
    using identical parameters, returning the three outputs as NumPy arrays.

    The Keras layer wraps the TensorFlow transform, whereas the NumPy frontend
    is an independent backend -- so agreement validates both the Keras wrapper
    and cross-backend numerical consistency.
    """
    N = x.shape[-1]
    inputs = Input(shape=(N,))
    model = Model(inputs, TimeFrequencyScattering(**kwargs)(inputs))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    keras_out = model.predict(x)
    tf_out = np.asarray(TimeFrequencyScatteringTensorFlow(shape=(N,), **kwargs)(x))
    np_out = np.asarray(TimeFrequencyScatteringNumPy(shape=(N,), **kwargs)(x))
    return keras_out, tf_out, np_out


def _assert_frontends_agree(x, **kwargs):
    keras_out, tf_out, np_out = _jtfs_frontend_outputs(x, **kwargs)
    assert keras_out.shape == tf_out.shape == np_out.shape, (
        keras_out.shape, tf_out.shape, np_out.shape)
    assert np.allclose(keras_out, tf_out, rtol=1e-4, atol=1e-5), \
        "keras vs tensorflow max|diff|={}".format(np.abs(keras_out - tf_out).max())
    assert np.allclose(keras_out, np_out, rtol=1e-4, atol=1e-5), \
        "keras vs numpy max|diff|={}".format(np.abs(keras_out - np_out).max())
    return keras_out


def test_TimeFrequencyScattering_matches_tf_and_numpy_time():
    """The Keras ``format='time'`` output matches the TensorFlow and NumPy
    frontends, under both default (local) and global temporal averaging."""
    rng = np.random.RandomState(0)
    N = 2 ** 12
    x = rng.randn(2, N).astype('float32')
    _assert_frontends_agree(x, J=6, J_fr=2, Q=8, format='time')
    _assert_frontends_agree(x, J=6, J_fr=2, Q=8, T='global', format='time')


def test_TimeFrequencyScattering_matches_tf_and_numpy_joint():
    """The Keras ``format='joint'`` (4D) output matches the TensorFlow and
    NumPy frontends."""
    rng = np.random.RandomState(0)
    N = 2 ** 12
    x = rng.randn(2, N).astype('float32')
    _assert_frontends_agree(x, J=6, J_fr=2, Q=8, format='joint')
