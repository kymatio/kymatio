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

.. automodule:: kymatio.numpy
    :members:
    :show-inheritance:

Scikit-learn
============

.. automodule:: kymatio.sklearn
    :members:
    :show-inheritance:

PyTorch
=======

.. automodule:: kymatio.torch
    :members:
    :show-inheritance:

TensorFlow
==========

.. automodule:: kymatio.tensorflow
    :members:
    :show-inheritance:

Keras
=====

.. automodule:: kymatio.keras
    :members:
    :show-inheritance:
