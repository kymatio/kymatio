Documentation
*************

The Kymatio API is divided between different frontends, which perform the same operations, but integrate in different frameworks. This integration allows the user to take advantage of different features available in certain frameworks, such as autodifferentiation and GPU processing in PyTorch and TensorFlow/Keras, while having code that runs almost identically in NumPy or scikit-learn. The available frontends are:

* ``kymatio.numpy`` for NumPy,
* ``kymatio.sklearn`` for scikit-learn (as ``Transformer`` and ``Estimator`` objects),
* ``kymatio.torch`` for PyTorch,
* ``kymatio.tensorflow`` for TensorFlow, and
* ``kymatio.keras`` for Keras.

To instantiate a ``Scattering2D`` object for the NumPy frontend, run::

    from kymatio.numpy import Scattering2D
    S = Scattering2D(J=2, shape=(32, 32))

Alternatively, the object may be instantiated in a dynamic way using the ``kymatio.Scattering2D`` object by providing a ``frontend`` argument. This object then transforms itself to the desired frontend. Using this approach, the above example becomes::

    from kymatio import Scattering2D
    S = Scattering2D(J=2, shape=(32, 32), frontend='numpy')

In Kymatio 0.2, the default frontend is ``torch`` for backwards compatibility reasons, but this change to ``numpy`` in the next version.

NumPy
=====

The NumPy frontend takes ``ndarray``\s as input and outputs ``ndarray``\s. All computation is done on the CPU, which means that it will be slow for large inputs.

.. automodule:: kymatio.numpy
    :members:
    :show-inheritance:

Scikit-learn
============

The scikit-learn frontend is both a ``Transformer`` and an ``Estimator``, making it easy to integrate the object into a scikit-learn ``Pipeline``. For example, you can write the following::

    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    from kymatio.sklearn import Scattering2D

    S = Scattering2D(J=1, shape=(8, 8))
    classifier = LogisticRegression()
    pipeline = Pipeline([('scatter', S), ('clf', classifier)])

which creates a ``Pipeline`` consisting of a 2D scattering transform and a logistic regression estimator.


.. automodule:: kymatio.sklearn
    :members:
    :show-inheritance:

PyTorch
=======

The PyTorch frontend inherits from ``torch.nn.Module``. As a result, it can be integrated with other PyTorch ``Module``\s to create a computational model. It also supports the `cuda`, `cpu`, and `to` methods, allowing the user to easily move the object from CPU to GPU and back.

.. automodule:: kymatio.torch
    :members:
    :show-inheritance:

TensorFlow
==========

The TensorFlow frontend inherits from ``tf.Module``. It therefore supports the same functionality and can be integrated with other ``Module``\s to form a computational graph.

.. automodule:: kymatio.tensorflow
    :members:
    :show-inheritance:

Keras
=====

The Keras frontend extends the TensorFlow frontend by implementing the scattering transform as a Keras ``Layer``. This can be combined with other ``Layer``\s to form a ``Model`` that may then be trained for a given task. Note that since Keras infers the input shape of a ``Layer``, we do not specify the shape when creating the scattering object. The result may look something like::

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Flatten, Dense

    from kymatio.keras import Scattering2D

    in_layer = Input(shape=(28, 28))
    sc = Scattering2D(J=3)(in_layer)
    sc_flat = Flatten()(sc)
    out_layer = Dense(10, activation='softmax')(sc_flat)

    model = Model(in_layer, out_layer)

where we feed the scattering coefficients into a dense layer with ten outputs for handwritten digit classification on MNIST.

.. automodule:: kymatio.keras
    :members:
    :show-inheritance:
